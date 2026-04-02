#!/usr/bin/env python3
"""
Naive RAG Evaluation Pipeline
==============================
Integrated end-to-end evaluation of RAG quality across ANN indices.

Evaluation LLMs (via OpenRouter): DeepSeek V3.2, GPT-5-mini
Judge LLMs (via OpenRouter):      Gemini 2.0 Flash, Kimi K2

Usage:
    # 1. Prepare QA pairs only (no API key needed)
    python naive_rag_pipeline.py --prepare-only

    # 2. Full pipeline
    python naive_rag_pipeline.py --openrouter-key <KEY>

    # 3. Evaluate specific index families
    python naive_rag_pipeline.py --openrouter-key <KEY> --indices hnsw ivf

    # 4. Pick specific configs by recall range
    python naive_rag_pipeline.py --openrouter-key <KEY> --min-recall 0.85 --max-recall 0.95
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

try:
    import aiohttp
except ImportError:
    print("aiohttp is required: pip install aiohttp")
    sys.exit(1)

try:
    import h5py
except ImportError:
    print("h5py is required: pip install h5py")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "results" / "neurips23" / "ood" / "msmarco-10M" / "10"
DEFAULT_GT_JSON = SCRIPT_DIR / "gt.json"
DEFAULT_GT_BIN = SCRIPT_DIR.parent / "data" / "msmarco" / "gt.bin"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "naive_rag"

# Index family name → results subdirectory
INDEX_DIRS = {
    "hnsw": "faiss_hnsw",
    "ivf": "faiss-ivf",
    "scann": "scann",
    "diskann": "diskann",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Eval models (OpenRouter)
EVAL_MODELS = {
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "gpt-5-mini": "openai/gpt-5-mini",
}

# Judge models (OpenRouter)
JUDGE_MODELS = {
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}

# Fixed index configurations (5 tiers, aligned by recall)
# Tier 1: ~0.85, Tier 2: ~0.87, Tier 3: ~0.90, Tier 4: ~0.91, Tier 5: ~0.94 (no HNSW)
SELECTED_CONFIGS = {
    "hnsw": [
        "ip_M_16_efConstruction_300_ef_200.hdf5",       # 0.851
        "ip_M_16_efConstruction_300_ef_300.hdf5",       # 0.872
        "ip_M_16_efConstruction_300_ef_500.hdf5",       # 0.893
        "ip_M_16_efConstruction_300_ef_1000.hdf5",      # 0.914
    ],
    "ivf": [
        "ip_n_list_4000_n_probe_30.hdf5",               # 0.862
        "ip_n_list_4000_n_probe_40.hdf5",               # 0.882
        "ip_n_list_4000_n_probe_60.hdf5",               # 0.903
        "ip_n_list_4000_n_probe_80.hdf5",               # 0.917
        "ip_n_list_4000_n_probe_150.hdf5",              # 0.945
    ],
    "scann": [
        "ip_dim_768_download_false_metric_ip_tree_size_10000_leaves_to_search_20_reorder_150.hdf5",   # 0.862
        "ip_dim_768_download_false_metric_ip_tree_size_10000_leaves_to_search_30_reorder_150.hdf5",   # 0.883
        "ip_dim_768_download_false_metric_ip_tree_size_10000_leaves_to_search_40_reorder_150.hdf5",   # 0.896
        "ip_dim_768_download_false_metric_ip_tree_size_10000_leaves_to_search_60_reorder_150.hdf5",   # 0.914
        "ip_dim_768_download_false_metric_ip_tree_size_10000_leaves_to_search_100_reorder_150.hdf5",  # 0.933
    ],
    "diskann": [
        "ip_L_500_R_32_buildthreads_64_Ls_15_T_64.hdf5",  # 0.877
        "ip_L_300_R_16_buildthreads_64_Ls_40_T_64.hdf5",  # 0.886
        "ip_L_500_R_32_buildthreads_64_Ls_20_T_64.hdf5",  # 0.903
        "ip_L_300_R_16_buildthreads_64_Ls_60_T_64.hdf5",  # 0.914
        "ip_L_500_R_32_buildthreads_64_Ls_40_T_64.hdf5",  # 0.947
    ],
}

# ═══════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════

RAG_SYSTEM_PROMPT = (
    "You are an accurate and reliable AI assistant that can answer questions "
    "purely relying on external documents. "
    "Please note that external documents may contain noisy or factually incorrect information. "
    "If the information in the document contains the correct answer, you will give an accurate answer. "
    "If the information in the document does not contain the answer, you will generate "
    "'I can not answer the question because of the insufficient information in documents.', "
    "never use your local knowledge. "
    'Please output your answer as JSON: {"Answer": "<your answer>"}'
)

RAG_USER_TEMPLATE = "Document:\n{DOCS}\n\nQuestion:\n{QUERY}"

JUDGE_SYSTEM_PROMPT = (
    "You are an expert answer evaluator. Given a question, a reference (ground truth) answer, "
    "and a predicted answer, determine whether the prediction is correct.\n\n"
    "Rules:\n"
    '- "correct": The prediction conveys the same core meaning as the reference, even if phrased differently.\n'
    '- "incorrect": The prediction is wrong, irrelevant, or contradicts the reference.\n'
    '- "insufficient": The model explicitly stated it cannot answer due to insufficient information.\n\n'
    "Be lenient with paraphrasing but strict with factual accuracy.\n"
    'Output exactly: {"verdict": "correct"|"incorrect"|"insufficient", "explanation": "<brief reason>"}'
)

JUDGE_USER_TEMPLATE = (
    "Question: {QUERY}\n\n"
    "Reference Answer: {GT_ANSWER}\n\n"
    "Predicted Answer: {PREDICTION}"
)


# ═══════════════════════════════════════════════════════════════
# Async LLM Client
# ═══════════════════════════════════════════════════════════════

class AsyncLLMClient:
    """OpenAI-compatible async client with concurrency control and retries."""

    def __init__(self, api_key: str, base_url: str = OPENROUTER_URL,
                 max_concurrent: int = 10, max_retries: int = 6):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._request_count = 0
        self._error_count = 0

    async def generate(self, session: aiohttp.ClientSession, model_id: str,
                       system_prompt: str, user_prompt: str,
                       temperature: float = 0.2) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        self.base_url, headers=headers, json=payload,
                        timeout=aiohttp.ClientTimeout(total=90),
                    ) as resp:
                        body = await resp.json()

                        if resp.status == 429 or resp.status >= 500:
                            wait = min(2 ** attempt + random.uniform(0.5, 1.5), 60)
                            logger.warning(
                                f"HTTP {resp.status} from {model_id} (attempt {attempt+1}/{self.max_retries}), "
                                f"retrying in {wait:.1f}s"
                            )
                            await asyncio.sleep(wait)
                            continue

                        if resp.status != 200:
                            self._error_count += 1
                            raise RuntimeError(
                                f"API error {resp.status} from {model_id}: "
                                f"{json.dumps(body, indent=2)[:500]}"
                            )

                        if "choices" not in body or not body["choices"]:
                            self._error_count += 1
                            raise RuntimeError(f"Empty response from {model_id}: {body}")

                        self._request_count += 1
                        return body["choices"][0]["message"]["content"]

                except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                    wait = min(2 ** attempt + random.uniform(0.5, 1.5), 60)
                    logger.warning(
                        f"{type(exc).__name__} for {model_id} (attempt {attempt+1}/{self.max_retries}), "
                        f"retrying in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)

            self._error_count += 1
            raise RuntimeError(f"Failed after {self.max_retries} retries for {model_id}")

    @property
    def stats(self) -> str:
        return f"{self._request_count} OK, {self._error_count} errors"


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_ground_truth(gt_json_path: Path, gt_bin_path: Path):
    """Load GT queries/answers and exact KNN neighbor indices."""
    queries = []
    with open(gt_json_path) as f:
        for line in f:
            queries.append(json.loads(line))

    n, d = map(int, np.fromfile(gt_bin_path, dtype="uint32", count=2))
    with open(gt_bin_path, "rb") as f:
        f.seek(8)
        I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)

    logger.info(f"Loaded {len(queries)} GT queries, KNN index shape {I.shape}")
    return queries, I


def load_selected_configs(results_dir: Path, families: List[str]) -> List[dict]:
    """Load the fixed set of aligned index configurations."""
    configs = []
    for family in families:
        dir_name = INDEX_DIRS.get(family, family)
        family_dir = results_dir / dir_name
        if not family_dir.is_dir():
            logger.warning(f"Directory not found: {family_dir}")
            continue

        filenames = SELECTED_CONFIGS.get(family, [])
        for fname in filenames:
            hdf5 = family_dir / fname
            if not hdf5.exists():
                logger.warning(f"Missing HDF5: {hdf5}")
                continue
            with h5py.File(hdf5, "r") as h:
                avg_recall = float(np.mean(h["metrics"]["knn"]["recalls"][:])) / 10.0
            configs.append({
                "name": f"{family}/{hdf5.stem}",
                "family": family,
                "path": str(hdf5),
                "avg_recall": round(avg_recall, 4),
            })

    configs.sort(key=lambda c: (c["family"], c["avg_recall"]))
    return configs


def collect_all_doc_ids(queries: List[dict], gt_knn: np.ndarray,
                        configs: List[dict], k: int = 10) -> set:
    """Pre-collect all document IDs needed across KNN + all index configs."""
    qids = [int(q["id"]) for q in queries]
    doc_ids = set()

    # KNN baseline
    for qid in qids:
        doc_ids.update(int(x) for x in gt_knn[qid][:k])

    # Each index config
    for cfg in configs:
        with h5py.File(cfg["path"], "r") as h:
            neighbors = h["neighbors"][:]
        for qid in qids:
            doc_ids.update(int(x) for x in neighbors[qid][:k])

    return doc_ids


def load_corpus_subset(doc_ids: set) -> Dict[int, str]:
    """Load only the needed documents from MSMARCO corpus via batch select."""
    from datasets import load_dataset
    logger.info(f"Loading {len(doc_ids)} documents from MSMARCO corpus...")
    ds = load_dataset("namespace-PT/msmarco-corpus", split="train")

    sorted_ids = sorted(doc_ids)
    # Batch select is much faster than individual indexing
    subset = ds.select(sorted_ids)
    texts = subset["content"]

    doc_map = {}
    for idx, doc_id in enumerate(sorted_ids):
        text = texts[idx]
        if len(text) > 512:
            text = text[:512]
        doc_map[doc_id] = text

    logger.info(f"Loaded {len(doc_map)} unique documents")
    return doc_map


def build_qa_pairs(queries: List[dict], neighbors: np.ndarray,
                   recalls: Optional[np.ndarray], doc_map: Dict[int, str],
                   k: int = 10) -> List[dict]:
    """Build QA pairs from neighbor arrays using pre-fetched doc texts."""
    pairs = []
    for q in queries:
        qid = int(q["id"])
        nbrs = neighbors[qid][:k]
        docs = [doc_map.get(int(did), "") for did in nbrs]

        pair = {
            "id": qid,
            "query": q["query"],
            "answer": q["answer"],
            "doc": docs,
        }
        if recalls is not None:
            pair["recall"] = round(float(recalls[qid]) / k, 4)
        else:
            pair["recall"] = 1.0  # exact KNN
        pairs.append(pair)

    return pairs


def save_pairs(pairs: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def load_pairs(path: Path) -> List[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


# ═══════════════════════════════════════════════════════════════
# Evaluation (async)
# ═══════════════════════════════════════════════════════════════

def _load_cache(path: Path) -> Dict[int, dict]:
    """Load JSONL cache, keyed by query ID."""
    cache = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                cache[entry["id"]] = entry
    return cache


def _save_cache(results: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def _eval_one(client: AsyncLLMClient, session: aiohttp.ClientSession,
                    model_id: str, pair: dict) -> dict:
    """Evaluate one QA pair."""
    docs_text = "\n".join(pair["doc"])
    prompt = RAG_USER_TEMPLATE.format(DOCS=docs_text, QUERY=pair["query"])

    try:
        raw = await client.generate(session, model_id, RAG_SYSTEM_PROMPT, prompt)
        # Try to parse JSON answer
        try:
            parsed = json.loads(raw)
            prediction = parsed.get("Answer", raw)
        except json.JSONDecodeError:
            # Sometimes model wraps in markdown code block
            stripped = raw.strip()
            if stripped.startswith("```"):
                stripped = "\n".join(stripped.split("\n")[1:])
                if stripped.endswith("```"):
                    stripped = stripped[:-3].strip()
                try:
                    parsed = json.loads(stripped)
                    prediction = parsed.get("Answer", raw)
                except json.JSONDecodeError:
                    prediction = raw
            else:
                prediction = raw

        return {
            "id": pair["id"],
            "query": pair["query"],
            "answer": pair["answer"],
            "recall": pair["recall"],
            "prediction": prediction,
            "raw_response": raw,
            "status": "ok",
        }
    except Exception as e:
        logger.error(f"Eval error for query {pair['id']}: {e}")
        return {
            "id": pair["id"],
            "query": pair["query"],
            "answer": pair["answer"],
            "recall": pair["recall"],
            "prediction": None,
            "raw_response": None,
            "status": "error",
            "error": str(e),
        }


async def evaluate_config(client: AsyncLLMClient, model_name: str, model_id: str,
                          qa_pairs: List[dict], cache_path: Path,
                          batch_size: int = 40) -> List[dict]:
    """Evaluate all QA pairs for one index config with one eval model."""
    cached = _load_cache(cache_path)
    to_eval = [p for p in qa_pairs if p["id"] not in cached]

    logger.info(f"    {model_name}: {len(cached)} cached, {len(to_eval)} remaining")

    if not to_eval:
        return [cached[p["id"]] for p in qa_pairs if p["id"] in cached]

    results = list(cached.values())
    connector = aiohttp.TCPConnector(limit=batch_size, limit_per_host=batch_size)

    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(to_eval), batch_size):
            batch = to_eval[i : i + batch_size]
            tasks = [_eval_one(client, session, model_id, p) for p in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in batch_results:
                if isinstance(r, BaseException):
                    logger.error(f"    Unhandled exception: {r}")
                    continue
                results.append(r)

            # Checkpoint after each batch
            _save_cache(results, cache_path)
            done = min(i + batch_size, len(to_eval))
            ok = sum(1 for r in results if r.get("status") == "ok")
            logger.info(f"    {model_name}: {done}/{len(to_eval)} sent, {ok} OK total")

    return results


# ═══════════════════════════════════════════════════════════════
# Judging (async)
# ═══════════════════════════════════════════════════════════════

async def _judge_one(client: AsyncLLMClient, session: aiohttp.ClientSession,
                     model_id: str, pred: dict) -> dict:
    """Judge one prediction against ground truth."""
    if pred.get("status") != "ok" or pred.get("prediction") is None:
        return {"id": pred["id"], "verdict": "error", "explanation": "no prediction"}

    gt_answer = pred["answer"]
    if isinstance(gt_answer, list):
        gt_answer = " | ".join(str(a) for a in gt_answer)

    prompt = JUDGE_USER_TEMPLATE.format(
        QUERY=pred["query"],
        GT_ANSWER=gt_answer,
        PREDICTION=pred["prediction"],
    )

    try:
        raw = await client.generate(session, model_id, JUDGE_SYSTEM_PROMPT, prompt)
        try:
            parsed = json.loads(raw)
            return {
                "id": pred["id"],
                "verdict": parsed.get("verdict", "unknown"),
                "explanation": parsed.get("explanation", ""),
            }
        except json.JSONDecodeError:
            # Fallback: extract verdict from text
            stripped = raw.strip()
            if stripped.startswith("```"):
                stripped = "\n".join(stripped.split("\n")[1:])
                if stripped.endswith("```"):
                    stripped = stripped[:-3].strip()
                try:
                    parsed = json.loads(stripped)
                    return {
                        "id": pred["id"],
                        "verdict": parsed.get("verdict", "unknown"),
                        "explanation": parsed.get("explanation", ""),
                    }
                except json.JSONDecodeError:
                    pass
            low = raw.lower()
            if "incorrect" in low:
                v = "incorrect"
            elif "correct" in low:
                v = "correct"
            elif "insufficient" in low:
                v = "insufficient"
            else:
                v = "unknown"
            return {"id": pred["id"], "verdict": v, "explanation": raw[:200]}
    except Exception as e:
        logger.error(f"Judge error for query {pred['id']}: {e}")
        return {"id": pred["id"], "verdict": "error", "explanation": str(e)}


async def judge_config(client: AsyncLLMClient, judge_name: str, model_id: str,
                       predictions: List[dict], cache_path: Path,
                       batch_size: int = 40) -> List[dict]:
    """Judge all predictions with one judge model."""
    cached = _load_cache(cache_path)
    to_judge = [p for p in predictions if p["id"] not in cached]

    logger.info(f"    {judge_name}: {len(cached)} cached, {len(to_judge)} remaining")

    if not to_judge:
        return [cached[p["id"]] for p in predictions if p["id"] in cached]

    results = list(cached.values())
    connector = aiohttp.TCPConnector(limit=batch_size, limit_per_host=batch_size)

    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(to_judge), batch_size):
            batch = to_judge[i : i + batch_size]
            tasks = [_judge_one(client, session, model_id, p) for p in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in batch_results:
                if isinstance(r, BaseException):
                    logger.error(f"    Unhandled exception: {r}")
                    continue
                results.append(r)

            _save_cache(results, cache_path)
            done = min(i + batch_size, len(to_judge))
            logger.info(f"    {judge_name}: {done}/{len(to_judge)} judged")

    return results


def cross_check_judges(j1: List[dict], j2: List[dict],
                       name1: str, name2: str) -> List[dict]:
    """Cross-check two judges. If they disagree, use the more conservative verdict."""
    m1 = {r["id"]: r for r in j1}
    m2 = {r["id"]: r for r in j2}
    all_ids = sorted(set(m1) | set(m2))

    merged = []
    agree = disagree = 0

    for qid in all_ids:
        v1 = m1.get(qid, {}).get("verdict", "missing")
        v2 = m2.get(qid, {}).get("verdict", "missing")

        if v1 == v2:
            final = v1
            agree += 1
        else:
            disagree += 1
            # Conservative tiebreak
            priority = {"error": 0, "incorrect": 1, "insufficient": 2, "unknown": 3, "correct": 4, "missing": 5}
            final = v1 if priority.get(v1, 5) <= priority.get(v2, 5) else v2

        merged.append({
            "id": qid,
            f"verdict_{name1}": v1,
            f"verdict_{name2}": v2,
            "final_verdict": final,
        })

    total = agree + disagree
    pct = agree / total * 100 if total else 0
    logger.info(f"    Judge agreement: {agree}/{total} ({pct:.1f}%)")
    return merged


# ═══════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════

def generate_report(all_results: dict, output_dir: Path):
    """Print and save summary report."""
    lines = ["", "=" * 72, "  NAIVE RAG EVALUATION REPORT", "=" * 72]

    summary_rows = []

    for config_name in sorted(all_results, key=lambda c: all_results[c].get("avg_recall", 0)):
        cr = all_results[config_name]
        avg_recall = cr.get("avg_recall", 0)
        lines.append(f"\n{'─' * 60}")
        lines.append(f"  Index: {config_name}  |  avg recall@10: {avg_recall:.3f}")
        lines.append(f"{'─' * 60}")

        for model_name, mr in cr.get("models", {}).items():
            judgments = mr.get("cross_checked", [])
            total = len(judgments)
            if total == 0:
                continue
            correct = sum(1 for j in judgments if j["final_verdict"] == "correct")
            incorrect = sum(1 for j in judgments if j["final_verdict"] == "incorrect")
            insuff = sum(1 for j in judgments if j["final_verdict"] == "insufficient")
            errors = total - correct - incorrect - insuff

            lines.append(
                f"  {model_name:20s}  correct={correct}/{total} ({correct/total*100:5.1f}%)  "
                f"incorrect={incorrect}  insufficient={insuff}  errors={errors}"
            )

            # Per-judge breakdown
            judge_names = list(JUDGE_MODELS.keys())
            for jn in judge_names:
                key = f"verdict_{jn}"
                jc = sum(1 for j in judgments if j.get(key) == "correct")
                lines.append(f"    {jn:30s}  correct={jc}/{total} ({jc/total*100:5.1f}%)")

            summary_rows.append({
                "index": config_name,
                "family": config_name.split("/")[0],
                "avg_recall": avg_recall,
                "eval_model": model_name,
                "total": total,
                "correct": correct,
                "incorrect": incorrect,
                "insufficient": insuff,
                "accuracy": round(correct / total, 4) if total else 0,
            })

    report = "\n".join(lines)
    print(report)

    # Save report
    (output_dir / "report.txt").write_text(report)

    # Save CSV summary
    if summary_rows:
        import csv
        csv_path = output_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        logger.info(f"Summary CSV: {csv_path}")

    logger.info(f"Report saved: {output_dir / 'report.txt'}")


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

async def run_pipeline(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    # ── 1. Load selected index configs ──────────────────────────
    logger.info("Step 1: Loading selected index configurations...")
    configs = load_selected_configs(results_dir, args.indices)
    logger.info(f"Found {len(configs)} configurations:")
    for c in configs:
        logger.info(f"  {c['name']:60s}  recall={c['avg_recall']:.3f}")

    if not configs:
        logger.error("No index configurations found. Check --results-dir and --indices.")
        return

    # ── 2. Load GT ─────────────────────────────────────────────
    logger.info("Step 2: Loading ground truth...")
    queries, gt_knn = load_ground_truth(Path(args.gt_json), Path(args.gt_bin))

    # ── 3. Check which QA pairs need generating ───────────────
    # Only load corpus if we actually need to generate new pairs
    pairs_to_generate = []  # (config_name, cfg_or_None)
    knn_cache = output_dir / "qa_pairs" / "knn.jsonl"
    if not knn_cache.exists():
        pairs_to_generate.append(("knn", None))

    for cfg in configs:
        safe = cfg["name"].replace("/", "__")
        if not (output_dir / "qa_pairs" / f"{safe}.jsonl").exists():
            pairs_to_generate.append((cfg["name"], cfg))

    # ── 4. Build QA pairs ──────────────────────────────────────
    logger.info("Step 4: Building QA pairs...")

    if pairs_to_generate:
        logger.info(f"  {len(pairs_to_generate)} configs need pair generation, collecting doc IDs...")
        all_doc_ids = collect_all_doc_ids(queries, gt_knn, configs, k=args.k)
        doc_map = load_corpus_subset(all_doc_ids)
    else:
        doc_map = None
        logger.info("  All QA pairs already cached")

    # KNN baseline
    if knn_cache.exists():
        knn_pairs = load_pairs(knn_cache)
        logger.info(f"  KNN baseline: loaded {len(knn_pairs)} cached pairs")
    else:
        knn_pairs = build_qa_pairs(queries, gt_knn, None, doc_map, k=args.k)
        save_pairs(knn_pairs, knn_cache)
        logger.info(f"  KNN baseline: generated {len(knn_pairs)} pairs")

    # Non-GT baseline (placeholder docs for control)
    non_gt_cache = output_dir / "qa_pairs" / "non_gt.jsonl"
    if not non_gt_cache.exists():
        non_gt_pairs = []
        placeholder = "This is a placeholder document with no relevant information."
        for q in queries:
            non_gt_pairs.append({
                "id": int(q["id"]),
                "query": q["query"],
                "answer": q["answer"],
                "recall": 0.0,
                "doc": [placeholder] * args.k,
            })
        save_pairs(non_gt_pairs, non_gt_cache)
        logger.info(f"  Non-GT control: generated {len(non_gt_pairs)} pairs")

    # Index configs
    config_pairs = {}
    for cfg in configs:
        safe = cfg["name"].replace("/", "__")
        pair_cache = output_dir / "qa_pairs" / f"{safe}.jsonl"
        if pair_cache.exists():
            pairs = load_pairs(pair_cache)
            logger.info(f"  {cfg['name']}: loaded {len(pairs)} cached pairs")
        else:
            with h5py.File(cfg["path"], "r") as h:
                neighbors = h["neighbors"][:]
                recalls = h["metrics"]["knn"]["recalls"][:]
            pairs = build_qa_pairs(queries, neighbors, recalls, doc_map, k=args.k)
            save_pairs(pairs, pair_cache)
            logger.info(f"  {cfg['name']}: generated {len(pairs)} pairs (recall={cfg['avg_recall']:.3f})")
        config_pairs[cfg["name"]] = (pairs, cfg)

    del doc_map  # Free memory

    if args.prepare_only:
        logger.info("Prepare-only mode: QA pairs generated. Exiting.")
        logger.info(f"Output directory: {output_dir}")
        return

    # ── 5. Evaluate + Judge ────────────────────────────────────
    if not args.openrouter_key:
        logger.error("No --openrouter-key provided. Use --prepare-only to generate QA pairs without API.")
        return

    client = AsyncLLMClient(
        args.openrouter_key,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
    )

    # Combine KNN + index configs for evaluation
    all_eval_targets = {"knn": (knn_pairs, {"name": "knn", "avg_recall": 1.0})}
    all_eval_targets.update(config_pairs)

    all_results = {}

    for config_name, (qa_pairs, cfg) in all_eval_targets.items():
        safe = config_name.replace("/", "__")
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Evaluating: {config_name} ({len(qa_pairs)} queries, recall={cfg['avg_recall']:.3f})")
        logger.info(f"{'=' * 50}")

        model_results = {}

        for model_name, model_id in EVAL_MODELS.items():
            logger.info(f"  Eval model: {model_name}")
            eval_cache = output_dir / "eval" / f"{safe}__{model_name}.jsonl"
            eval_cache.parent.mkdir(parents=True, exist_ok=True)

            predictions = await evaluate_config(
                client, model_name, model_id, qa_pairs, eval_cache,
                batch_size=args.batch_size,
            )

            ok_preds = [p for p in predictions if p.get("status") == "ok"]
            logger.info(f"    {model_name}: {len(ok_preds)}/{len(qa_pairs)} successful predictions")

            # Judge with both judges
            logger.info(f"  Judging {model_name} predictions...")
            judge_outputs = {}
            for judge_name, judge_id in JUDGE_MODELS.items():
                judge_cache = output_dir / "judge" / f"{safe}__{model_name}__{judge_name}.jsonl"
                judge_cache.parent.mkdir(parents=True, exist_ok=True)

                judgments = await judge_config(
                    client, judge_name, judge_id, predictions, judge_cache,
                    batch_size=args.batch_size,
                )
                judge_outputs[judge_name] = judgments

            # Cross-check
            jnames = list(JUDGE_MODELS.keys())
            cross = cross_check_judges(
                judge_outputs[jnames[0]], judge_outputs[jnames[1]],
                jnames[0], jnames[1],
            )

            cross_path = output_dir / "cross_check" / f"{safe}__{model_name}.jsonl"
            cross_path.parent.mkdir(parents=True, exist_ok=True)
            _save_cache(cross, cross_path)

            model_results[model_name] = {
                "predictions_count": len(ok_preds),
                "cross_checked": cross,
            }

        all_results[config_name] = {
            "avg_recall": cfg["avg_recall"],
            "models": model_results,
        }

    # ── 6. Report ──────────────────────────────────────────────
    logger.info("\nStep 6: Generating report...")
    generate_report(all_results, output_dir)
    logger.info(f"\nAPI stats: {client.stats}")
    logger.info(f"All results in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Naive RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Paths
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--gt-json", type=str, default=str(DEFAULT_GT_JSON))
    parser.add_argument("--gt-bin", type=str, default=str(DEFAULT_GT_BIN))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    # Index selection
    parser.add_argument("--indices", nargs="+", default=["hnsw", "ivf", "scann", "diskann"],
                        choices=list(INDEX_DIRS.keys()),
                        help="Index families to evaluate")
    # Recall range args removed — using fixed aligned configs from SELECTED_CONFIGS

    # API
    parser.add_argument("--openrouter-key", type=str,
                        default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")

    # Execution control
    parser.add_argument("--k", type=int, default=10, help="Number of retrieved docs per query")
    parser.add_argument("--max-concurrent", type=int, default=10,
                        help="Max concurrent API requests")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=40,
                        help="Async batch size (checkpoint after each batch)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only generate QA pairs, skip LLM evaluation")

    args = parser.parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
