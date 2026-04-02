#!/usr/bin/env python3
"""
Judge whether model answers are based on retrieved docs or internal knowledge.
Uses GPT-4o-mini via OpenRouter as the judge.

Usage:
    python judge_doc_validity.py --config hnsw_ef80
    python judge_doc_validity.py --config all
"""
import asyncio
import aiohttp
import json
import os
import sys
import random
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "openai/gpt-4o-mini"
API_KEY = os.environ.get("OPENROUTER_API_KEY")

DOC_VALIDITY_PROMPT = """You are judging whether an AI model's answer to a question was based on the retrieved documents or fabricated from internal knowledge.

Question: {question}
Ground Truth Answer: {gt_answer}
Model's Answer: {model_answer}

Retrieved Documents (from all search turns):
{documents}

Your task: Determine if the retrieved documents contain sufficient information to support the model's answer.

Rules:
- "doc_based": The documents clearly contain the information needed to produce the model's answer. The answer can be derived from the document content.
- "internal_knowledge": The documents do NOT contain the information. The model must have used its own knowledge to answer. This includes cases where documents are about a related topic but don't contain the specific answer.
- "partial": The documents contain some supporting information but the model filled in crucial gaps with internal knowledge.

Output exactly: {{"verdict": "doc_based"|"internal_knowledge"|"partial", "reason": "<brief>"}}"""


async def judge_one(session, headers, trace, doc_texts, semaphore):
    """Judge one trace for doc validity."""
    qid = trace["question_id"]

    # Collect all retrieved doc texts
    all_docs = []
    for turn in trace.get("turns", []):
        if turn.get("action") == "search":
            for rid in turn.get("retrieved_ids", []):
                text = doc_texts.get(str(rid), "")
                if text and len(text) > 500:
                    text = text[:500]
                if text:
                    all_docs.append(text)

    if not all_docs:
        return {"id": qid, "verdict": "internal_knowledge", "reason": "no docs retrieved"}

    docs_text = "\n---\n".join(all_docs[:15])  # Cap at 15 docs
    gt = trace["answers"]
    gt_str = " | ".join(gt) if isinstance(gt, list) else str(gt)

    prompt = DOC_VALIDITY_PROMPT.format(
        question=trace["question"],
        gt_answer=gt_str,
        model_answer=trace.get("final_answer", ""),
        documents=docs_text,
    )

    payload = {
        "model": JUDGE_MODEL,
        "temperature": 0.1,
        "max_tokens": 150,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with semaphore:
        for attempt in range(5):
            try:
                async with session.post(
                    OPENROUTER_URL, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                        continue
                    data = await resp.json()
                    raw = data["choices"][0]["message"]["content"]
                    try:
                        parsed = json.loads(raw)
                        return {"id": qid, "verdict": parsed.get("verdict", "unknown"), "reason": parsed.get("reason", "")}
                    except json.JSONDecodeError:
                        # Try to extract from text
                        raw_l = raw.lower()
                        if "internal_knowledge" in raw_l:
                            v = "internal_knowledge"
                        elif "doc_based" in raw_l:
                            v = "doc_based"
                        elif "partial" in raw_l:
                            v = "partial"
                        else:
                            v = "unknown"
                        return {"id": qid, "verdict": v, "reason": raw[:100]}
            except Exception as e:
                await asyncio.sleep(2 ** attempt)
        return {"id": qid, "verdict": "error", "reason": "failed after retries"}


async def judge_config(config_name, output_dir):
    test_ids = set(json.load(open(output_dir / "test_ids.json")))

    # Load traces
    traces = []
    with open(output_dir / "traces" / f"{config_name}.jsonl") as f:
        for line in f:
            t = json.loads(line)
            if t["question_id"] in test_ids and t["status"] == "answered":
                traces.append(t)

    # Load existing correctness judgments
    gem = {}
    gem_path = output_dir / "judgments" / f"{config_name}__gemini-2.0-flash.jsonl"
    if gem_path.exists():
        with open(gem_path) as f:
            for line in f:
                e = json.loads(line)
                gem[e["id"]] = e

    # Only judge correctly-answered traces (where Gemini said "correct")
    correct_traces = [t for t in traces if gem.get(t["question_id"], {}).get("verdict") == "correct"]
    # Also judge incorrect ones to get full picture
    incorrect_traces = [t for t in traces if gem.get(t["question_id"], {}).get("verdict") != "correct"]

    logger.info(f"{config_name}: {len(correct_traces)} correct, {len(incorrect_traces)} incorrect to judge")

    # Load doc texts
    doc_texts = json.load(open(output_dir / "doc_texts_lookup.json"))

    # Load cache
    cache_path = output_dir / "doc_validity" / f"{config_name}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    cached[e["id"]] = e

    all_traces = correct_traces + incorrect_traces
    to_judge = [t for t in all_traces if t["question_id"] not in cached]
    logger.info(f"  {len(cached)} cached, {len(to_judge)} to judge")

    if not to_judge:
        results = list(cached.values())
    else:
        results = list(cached.values())
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        semaphore = asyncio.Semaphore(20)
        batch_size = 80
        connector = aiohttp.TCPConnector(limit=20)

        async with aiohttp.ClientSession(connector=connector) as session:
            for i in range(0, len(to_judge), batch_size):
                batch = to_judge[i:i + batch_size]
                tasks = [judge_one(session, headers, t, doc_texts, semaphore) for t in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in batch_results:
                    if isinstance(r, BaseException):
                        logger.error(f"  Exception: {r}")
                        continue
                    results.append(r)

                with open(cache_path, "w") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")

                done = min(i + batch_size, len(to_judge))
                doc_based = sum(1 for r in results if r.get("verdict") == "doc_based")
                logger.info(f"  {config_name}: {done}/{len(to_judge)} judged, {doc_based} doc_based so far")

    # Summarize
    validity = {r["id"]: r for r in results}

    correct_doc = sum(1 for t in correct_traces if validity.get(t["question_id"], {}).get("verdict") == "doc_based")
    correct_internal = sum(1 for t in correct_traces if validity.get(t["question_id"], {}).get("verdict") == "internal_knowledge")
    correct_partial = sum(1 for t in correct_traces if validity.get(t["question_id"], {}).get("verdict") == "partial")

    incorrect_doc = sum(1 for t in incorrect_traces if validity.get(t["question_id"], {}).get("verdict") == "doc_based")
    incorrect_internal = sum(1 for t in incorrect_traces if validity.get(t["question_id"], {}).get("verdict") == "internal_knowledge")

    logger.info(f"\n  {config_name} summary:")
    logger.info(f"    Correct + doc_based:     {correct_doc}")
    logger.info(f"    Correct + internal_know: {correct_internal}")
    logger.info(f"    Correct + partial:       {correct_partial}")
    logger.info(f"    Incorrect + doc_based:   {incorrect_doc}")
    logger.info(f"    Incorrect + internal:    {incorrect_internal}")

    return {
        "config": config_name,
        "total_answered": len(traces),
        "correct": len(correct_traces),
        "correct_doc_based": correct_doc,
        "correct_internal": correct_internal,
        "correct_partial": correct_partial,
        "incorrect_doc_based": incorrect_doc,
        "incorrect_internal": incorrect_internal,
    }


async def main(args):
    output_dir = Path(args.output_dir)

    if args.config == "all":
        configs = ["knn_flat", "hnsw_ef30", "hnsw_ef40", "hnsw_ef80", "ivf_np60", "ivf_np100", "ivf_np200"]
    else:
        configs = [args.config]

    results = []
    for cfg in configs:
        r = await judge_config(cfg, output_dir)
        results.append(r)

    # Print final table
    print(f"\n{'Config':15s} {'Answered':>8s} {'Correct':>8s} {'Doc-Based':>10s} {'Internal':>9s} {'Partial':>8s} {'RealCorr%':>9s}")
    print("=" * 75)
    for r in results:
        real_correct_pct = r["correct_doc_based"] / r["total_answered"] * 100 if r["total_answered"] else 0
        print(f"{r['config']:15s} {r['total_answered']:>8d} {r['correct']:>8d} {r['correct_doc_based']:>10d} {r['correct_internal']:>9d} {r['correct_partial']:>8d} {real_correct_pct:>8.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="all")
    parser.add_argument("--output-dir", default="output/agentic_rag")
    asyncio.run(main(parser.parse_args()))
