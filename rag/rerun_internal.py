#!/usr/bin/env python3
"""
Re-run queries labeled as internal_knowledge/partial with a stricter prompt.
If model switches to 'insufficient information', it confirms internal knowledge usage.
"""
import asyncio
import aiohttp
import json
import os
import sys
import random
import logging
import numpy as np
import faiss
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from agentic_rag_pipeline import (
    load_hotpotqa, HOTPOTQA_DATA, FLAT_INDEX_PATH,
    E5Encoder, VLLM_MODEL, parse_search_query, parse_answer,
    format_search_results, HnswlibWrapper, HNSW_CONFIGS, IVF_CONFIGS,
    HNSW_DIR, IVF_DIR, AgenticTrace,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

STRICT_PROMPT = (
    "Answer the given question. "
    "CRITICAL RULE: You are FORBIDDEN from using ANY internal or prior knowledge. "
    "You have NO knowledge about the world. You know NOTHING except what is in the retrieved documents. "
    "You MUST search first. You MUST base your answer ENTIRELY on retrieved documents. "
    "If the retrieved documents do not explicitly contain the answer, you MUST respond with "
    "'insufficient information' — do NOT guess, infer, or use any knowledge you have from training. "
    "Even if you think you know the answer, if it's not in the documents, say 'insufficient information'. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you need more knowledge, call the search engine by "
    "<search> query </search> and it will return results between "
    "<information> and </information>. "
    "Provide the final answer inside <answer> and </answer>.\n"
    "Question: {question}"
)


async def run_strict_episode(session, vllm_url, model_id, question, faiss_index, corpus, encoder, topk=5):
    """Run one episode with the strict prompt."""
    trace = AgenticTrace(
        question_id=question["id"],
        original_id=question.get("original_id", ""),
        answers=question["answers"],
        question=question["question"],
    )

    messages = [{"role": "user", "content": STRICT_PROMPT.format(question=question["question"])}]

    for turn in range(10):
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "stop": ["</search>"],
        }

        data = None
        for attempt in range(5):
            try:
                async with session.post(
                    f"{vllm_url}/chat/completions", json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        await asyncio.sleep(2 ** attempt + random.uniform(0.5, 1.5))
                        continue
                    if resp.status != 200:
                        trace.status = "error"
                        return trace
                    data = await resp.json()
                    break
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                await asyncio.sleep(2 ** attempt + random.uniform(0.5, 1.5))

        if data is None:
            trace.status = "error"
            return trace

        text = data["choices"][0]["message"]["content"]
        finish = data["choices"][0].get("finish_reason", "")

        full = text + "</search>" if finish == "stop" and "</search>" not in text else text

        answer = parse_answer(full)
        if answer:
            trace.final_answer = answer
            trace.status = "answered"
            trace.turns.append({"turn": turn, "action": "answer", "answer": answer, "response": full})
            return trace

        query = parse_search_query(full)
        if not query:
            trace.final_answer = text.strip()
            trace.status = "answered"
            trace.turns.append({"turn": turn, "action": "direct_answer", "response": full})
            return trace

        trace.num_searches += 1
        qvec = encoder.encode([query])
        dists, ids = faiss_index.search(qvec, topk)

        docs = []
        for idx in ids[0]:
            if 0 <= idx < len(corpus):
                content = corpus[idx].get("contents", "")
                if len(content) > 800:
                    content = content[:800]
                docs.append(content)

        info = format_search_results(docs, topk)
        trace.turns.append({
            "turn": turn, "action": "search", "search_query": query,
            "retrieved_ids": ids[0].tolist(),
        })
        messages.append({"role": "assistant", "content": full})
        messages.append({"role": "user", "content": info})

    trace.status = "max_turns"
    return trace


async def main():
    output_dir = Path("output/agentic_rag")

    # Load queries to rerun
    rerun_qids = set(json.load(open(output_dir / "rerun_internal_qids.json")))
    logger.info(f"Queries to rerun: {len(rerun_qids)}")

    # Load all questions
    questions = load_hotpotqa(HOTPOTQA_DATA, max_questions=99999)
    questions = [q for q in questions if q["id"] in rerun_qids]
    logger.info(f"Matched questions: {len(questions)}")

    # Load corpus
    logger.info("Loading corpus...")
    corpus = []
    with open(str(Path(__file__).resolve().parent.parent / "data" / "search-r1" / "wiki-18-corpus.jsonl")) as f:
        for line in f:
            corpus.append(json.loads(line))
    logger.info(f"Corpus: {len(corpus)}")

    # Load encoder
    encoder = E5Encoder(device="cuda:2")

    # Load KNN index (GPU) — rerun with perfect retrieval + strict prompt
    logger.info("Loading flat index on GPU...")
    flat = faiss.read_index(str(FLAT_INDEX_PATH))
    res = faiss.StandardGpuResources()
    flat_gpu = faiss.index_cpu_to_gpu(res, 3, flat)
    del flat

    # Run with strict prompt
    cache_path = output_dir / "traces" / "knn_strict_rerun.jsonl"
    cached = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    cached[t["question_id"]] = t

    to_run = [q for q in questions if q["id"] not in cached]
    logger.info(f"To run: {len(to_run)}, cached: {len(cached)}")

    results = list(cached.values())
    semaphore = asyncio.Semaphore(8)
    connector = aiohttp.TCPConnector(limit=8)

    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(to_run), 16):
            batch = to_run[i:i+16]

            async def run_one(q):
                async with semaphore:
                    return await run_strict_episode(
                        session, "http://localhost:8001/v1", VLLM_MODEL,
                        q, flat_gpu, corpus, encoder, topk=5,
                    )

            batch_results = await asyncio.gather(*[run_one(q) for q in batch], return_exceptions=True)
            for r in batch_results:
                if isinstance(r, BaseException):
                    logger.error(f"Exception: {r}")
                    continue
                results.append(r.to_dict())

            with open(cache_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            done = min(i + 16, len(to_run))
            answered = sum(1 for r in results if r.get("status") == "answered")
            logger.info(f"  {done}/{len(to_run)} done, {answered} answered")

    # ── Run same queries on HNSW and IVF configs ──
    import hnswlib

    index_configs = [
        ("hnsw_ef30", "hnsw", 30),
        ("hnsw_ef40", "hnsw", 40),
        ("hnsw_ef80", "hnsw", 80),
        ("ivf_np60",  "ivf",  60),
        ("ivf_np100", "ivf",  100),
        ("ivf_np200", "ivf",  200),
    ]

    hnsw_path = HNSW_DIR / f"e5_HNSW{HNSW_CONFIGS['M']}.hnswlib"
    ivf_path = IVF_DIR / f"e5_IVF{IVF_CONFIGS['n_list']}.index"

    hnsw_index = None
    ivf_index = None

    if hnsw_path.exists():
        logger.info("Loading HNSW index...")
        hnsw_index = hnswlib.Index(space="ip", dim=768)
        hnsw_index.load_index(str(hnsw_path))

    if ivf_path.exists():
        logger.info("Loading IVF index...")
        ivf_index = faiss.read_index(str(ivf_path))

    for config_name, idx_type, param in index_configs:
        if idx_type == "hnsw" and hnsw_index is None:
            continue
        if idx_type == "ivf" and ivf_index is None:
            continue

        if idx_type == "hnsw":
            hnsw_index.set_ef(param)
            search_index = HnswlibWrapper.__new__(HnswlibWrapper)
            search_index.index = hnsw_index
            search_index.search = lambda q, k, _idx=hnsw_index: (lambda l, d: (d, l))(*_idx.knn_query(q, k=k))
        else:
            ivf_index.nprobe = param
            search_index = ivf_index

        cfg_cache = output_dir / "traces" / f"{config_name}_strict_rerun.jsonl"
        cfg_cached = {}
        if cfg_cache.exists():
            with open(cfg_cache) as f:
                for line in f:
                    if line.strip():
                        t = json.loads(line)
                        cfg_cached[t["question_id"]] = t

        cfg_to_run = [q for q in questions if q["id"] not in cfg_cached]
        logger.info(f"\n{config_name}: {len(cfg_to_run)} to run, {len(cfg_cached)} cached")

        cfg_results = list(cfg_cached.values())
        connector2 = aiohttp.TCPConnector(limit=8)
        async with aiohttp.ClientSession(connector=connector2) as session:
            for i in range(0, len(cfg_to_run), 16):
                batch = cfg_to_run[i:i+16]

                async def run_one_cfg(q, si=search_index):
                    async with semaphore:
                        return await run_strict_episode(
                            session, "http://localhost:8001/v1", VLLM_MODEL,
                            q, si, corpus, encoder, topk=5,
                        )

                batch_results = await asyncio.gather(*[run_one_cfg(q) for q in batch], return_exceptions=True)
                for r in batch_results:
                    if isinstance(r, BaseException):
                        logger.error(f"Exception: {r}")
                        continue
                    cfg_results.append(r.to_dict())

                with open(cfg_cache, "w") as f:
                    for r in cfg_results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

                done = min(i + 16, len(cfg_to_run))
                answered = sum(1 for r in cfg_results if r.get("status") == "answered")
                logger.info(f"  {config_name}: {done}/{len(cfg_to_run)} done, {answered} answered")

    # ── Compare all configs ──
    import re
    def normalize(s):
        import string
        s = s.lower().strip()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = s.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(s.split())

    logger.info("\n" + "=" * 70)
    logger.info("STRICT PROMPT RESULTS (queries previously labeled internal/partial)")
    logger.info("=" * 70)

    all_configs = [("knn_strict_rerun", "KNN (strict)")] + \
                  [(f"{n}_strict_rerun", n) for n, _, _ in index_configs]

    logger.info(f"{'Config':20s} {'Total':>6s} {'Correct':>8s} {'Insuff':>8s} {'Wrong':>8s} {'Corr%':>7s}")
    logger.info("-" * 60)

    for trace_name, label in all_configs:
        trace_path = output_dir / "traces" / f"{trace_name}.jsonl"
        if not trace_path.exists():
            continue
        traces = {}
        with open(trace_path) as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    traces[t["question_id"]] = t

        correct = insuff = wrong = total = 0
        for qid in rerun_qids:
            if qid not in traces or traces[qid].get("status") != "answered":
                continue
            total += 1
            ans = (traces[qid].get("final_answer") or "").strip().lower()
            gts = traces[qid].get("answers", [])

            if "insufficient" in ans:
                insuff += 1
            elif any(normalize(g) == normalize(ans) for g in gts):
                correct += 1
            else:
                wrong += 1

        corr_pct = correct / total * 100 if total else 0
        logger.info(f"{label:20s} {total:>6d} {correct:>8d} {insuff:>8d} {wrong:>8d} {corr_pct:>6.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
