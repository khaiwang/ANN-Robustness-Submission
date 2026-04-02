#!/usr/bin/env python3
"""
Agentic RAG Evaluation Pipeline
================================
Search-R1 style multi-hop QA over Wikipedia using Qwen3-30B (vLLM)
with FAISS HNSW/IVF retrieval and full intermediate logging.

Adapted from Search-R1 (Apache 2.0): https://github.com/petergriffinjin/search-r1

Usage:
    # 1. Start vLLM server (separate terminal):
    #    python -m vllm.entrypoints.openai.api_server \
    #        --model Qwen/Qwen3-30B-A3B --dtype bfloat16 \
    #        --tensor-parallel-size 2 --port 8001

    # 2. Prepare indices (one-time):
    python agentic_rag_pipeline.py prepare-index

    # 3. Run evaluation:
    python agentic_rag_pipeline.py run --vllm-url http://localhost:8001/v1

    # 4. Judge results:
    python agentic_rag_pipeline.py judge --openrouter-key <KEY>
"""

import asyncio
import aiohttp
import json
import logging
import os
import re
import sys
import time
import random
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

try:
    import faiss
except ImportError:
    print("faiss is required: pip install faiss-gpu-cu12")
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
DATA_DIR = SCRIPT_DIR.parent / "data" / "search-r1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "agentic_rag"

# Paths
FLAT_INDEX_PATH = DATA_DIR / "e5_Flat.index"
CORPUS_PATH = DATA_DIR / "wiki-18-corpus.jsonl"
EMBEDDING_PATH = DATA_DIR / "corpus_embedding.bin"
GT_PATH = DATA_DIR / "gt.bin"
HOTPOTQA_DATA = DATA_DIR / "hotpotqa_data"
HNSW_DIR = DATA_DIR / "hnsw"
IVF_DIR = DATA_DIR / "ivf"

# Index configs matching paper: recall@5 range 0.8-0.9
# HNSW: M=64, efConstruction=500, sweep efSearch
HNSW_CONFIGS = {
    "M": 64,
    "efConstruction": 300,
    # Tuned on 1200 sampled questions: recall@5 ≈ 0.83, 0.86, 0.91
    "efSearch_values": [30, 40, 80],
}
# IVF: n_list=10000
IVF_CONFIGS = {
    "n_list": 10000,
    # Tuned on 1200 sampled questions: recall@5 ≈ 0.81, 0.85, 0.91
    "n_probe_values": [60, 100, 200],
}

K = 5  # top-K retrieval for agentic RAG

# Model
VLLM_MODEL = "Qwen/Qwen3-30B-A3B"
VLLM_DEFAULT_URL = "http://localhost:8001/v1"

# Judge models (same as naive RAG)
JUDGE_MODELS = {
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# E5 query encoder
E5_MODEL = "intfloat/e5-base-v2"

# ═══════════════════════════════════════════════════════════════
# Search-R1 Prompt (from infer.py)
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "Answer the given question. "
    "You MUST NOT use your internal knowledge to answer the question. "
    "You MUST search for information first using the search engine before answering. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you need more knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as you want. "
    "You must base your answer ONLY on the information retrieved from searches. "
    "Never use your own knowledge. If the retrieved information is insufficient, "
    "say 'insufficient information'. "
    "Provide the final answer inside <answer> and </answer>, without detailed explanation.\n"
    "Question: {question}"
)

MAX_TURNS = 10  # max search iterations per question

# ═══════════════════════════════════════════════════════════════
# Corpus & Index Utilities
# ═══════════════════════════════════════════════════════════════

class HnswlibWrapper:
    """Wraps hnswlib index with FAISS-compatible .search() interface."""

    def __init__(self, path: str, dim: int, ef: int = 100):
        import hnswlib
        self.index = hnswlib.Index(space="ip", dim=dim)
        self.index.load_index(path)
        self.index.set_ef(ef)

    def search(self, query: np.ndarray, k: int):
        labels, distances = self.index.knn_query(query, k=k)
        return distances, labels


def load_corpus(path: Path) -> List[dict]:
    """Load wiki-18 JSONL corpus."""
    logger.info(f"Loading corpus from {path}...")
    corpus = []
    with open(path) as f:
        for line in f:
            corpus.append(json.loads(line))
    logger.info(f"Corpus loaded: {len(corpus)} documents")
    return corpus


def load_embedding(path: Path) -> np.ndarray:
    """Load corpus embeddings from binary file."""
    n, d = map(int, np.fromfile(path, dtype="uint32", count=2))
    logger.info(f"Loading embeddings: n={n}, d={d}")
    with open(path, "rb") as f:
        f.seek(8)
        emb = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return emb


def save_gt(indices, distances, n, k, path):
    with open(path, "wb") as f:
        f.write(np.array([n, k], dtype=np.uint32).tobytes())
        f.write(np.array(indices, dtype=np.int32).tobytes())
        f.write(np.array(distances, dtype=np.float32).tobytes())


def read_gt(path):
    n, k = map(int, np.fromfile(path, dtype="uint32", count=2))
    with open(path, "rb") as f:
        f.seek(8)
        I = np.fromfile(f, dtype="int32", count=n * k).reshape(n, k)
        D = np.fromfile(f, dtype="float32", count=n * k).reshape(n, k)
    return I, D


def extract_embeddings_from_flat(flat_index_path: Path, out_path: Path):
    """Extract embeddings from flat index and save as .bin."""
    if out_path.exists():
        logger.info(f"Embeddings already exist: {out_path}")
        return
    logger.info(f"Extracting embeddings from {flat_index_path}...")
    flat_index = faiss.read_index(str(flat_index_path))
    n = flat_index.ntotal
    d = flat_index.d
    embeddings = np.zeros((n, d), dtype=np.float32)
    flat_index.reconstruct_n(0, n, embeddings)
    with open(out_path, "wb") as f:
        f.write(np.array([n, d], dtype=np.uint32).tobytes())
        f.write(embeddings.tobytes())
    logger.info(f"Saved embeddings: {n}x{d} to {out_path}")


def build_hnsw_index(embeddings: np.ndarray, M: int, efC: int, save_path: Path):
    """Build HNSW index."""
    if save_path.exists():
        logger.info(f"HNSW index exists: {save_path}")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building HNSW index: M={M}, efC={efC}, {embeddings.shape}...")
    faiss.omp_set_num_threads(64)
    index = faiss.IndexHNSWFlat(embeddings.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efC
    index.train(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(save_path))
    logger.info(f"HNSW index saved: {save_path}")


def build_ivf_index(embeddings: np.ndarray, n_list: int, save_path: Path, gpu_id: int = 2):
    """Build IVF index with GPU-accelerated k-means training."""
    if save_path.exists():
        logger.info(f"IVF index exists: {save_path}")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building IVF index: n_list={n_list}, {embeddings.shape} (GPU training on cuda:{gpu_id})...")

    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], n_list, faiss.METRIC_INNER_PRODUCT)

    # Train on GPU (k-means clustering is the expensive part)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index)
    train_data = embeddings[:500000]
    logger.info(f"  Training IVF on GPU {gpu_id} with {len(train_data)} vectors...")
    gpu_index.train(train_data)

    # Copy trained state back to CPU for adding + saving
    index = faiss.index_gpu_to_cpu(gpu_index)
    del gpu_index, res

    logger.info(f"  Adding {len(embeddings)} vectors (CPU)...")
    index.add(embeddings)
    faiss.write_index(index, str(save_path))
    logger.info(f"IVF index saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# E5 Query Encoder
# ═══════════════════════════════════════════════════════════════

class E5Encoder:
    """Encode queries using E5 model."""

    def __init__(self, model_path: str = E5_MODEL, device: str = "cuda:0"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.device = device

    @staticmethod
    def mean_pooling(output, attention_mask):
        import torch
        token_embeddings = output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        import torch
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # E5 requires "query: " prefix
            batch = [f"query: {t}" for t in batch]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**inputs)
            emb = self.mean_pooling(output, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# HotpotQA Data Loading
# ═══════════════════════════════════════════════════════════════

def load_hotpotqa(data_dir: Path, split: str = "test", max_questions: int = 2000) -> List[dict]:
    """Load HotpotQA questions from parquet."""
    import pyarrow.parquet as pq
    parquet_path = data_dir / f"{split}.parquet"
    if not parquet_path.exists():
        for p in data_dir.rglob(f"{split}.parquet"):
            parquet_path = p
            break
    if not parquet_path.exists():
        raise FileNotFoundError(f"No {split}.parquet found in {data_dir}")

    logger.info(f"Loading HotpotQA from {parquet_path}...")
    table = pq.read_table(parquet_path)

    # Fields: id, question, golden_answers, data_source, ...
    data_sources = table.column("data_source").to_pylist()
    questions_col = table.column("question").to_pylist()
    answers_col = table.column("golden_answers").to_pylist()
    ids_col = table.column("id").to_pylist()

    questions = []
    for i in range(table.num_rows):
        if data_sources[i] != "hotpotqa":
            continue
        answers = answers_col[i]
        if isinstance(answers, str):
            answers = [answers]
        questions.append({
            "id": len(questions),
            "original_id": ids_col[i],
            "question": questions_col[i],
            "answers": answers,
        })
        if len(questions) >= max_questions:
            break

    logger.info(f"Loaded {len(questions)} HotpotQA questions")
    return questions


# ═══════════════════════════════════════════════════════════════
# Agentic Loop
# ═══════════════════════════════════════════════════════════════

def parse_search_query(text: str) -> Optional[str]:
    """Extract search query from <search>...</search> tags."""
    match = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def format_search_results(docs: List[str], topk: int = 5) -> str:
    """Format retrieved documents as <information> block."""
    truncated = []
    for doc in docs[:topk]:
        if len(doc) > 800:
            doc = doc[:800]
        truncated.append(doc)
    return "<information>\n" + "\n\n".join(truncated) + "\n</information>\n"


@dataclass
class AgenticTrace:
    """Full trace of one agentic QA episode."""
    question_id: int
    original_id: str
    question: str
    answers: List[str]
    turns: List[dict] = field(default_factory=list)
    final_answer: Optional[str] = None
    num_searches: int = 0
    status: str = "pending"  # pending, answered, max_turns, error

    def to_dict(self):
        return asdict(self)


async def run_agentic_episode(
    session: aiohttp.ClientSession,
    vllm_url: str,
    model_id: str,
    question: dict,
    faiss_index,
    corpus: List[dict],
    encoder: "E5Encoder",
    max_turns: int = MAX_TURNS,
    topk: int = 5,
    temperature: float = 0.7,
) -> AgenticTrace:
    """Run one agentic QA episode: generate → search → observe → repeat."""
    trace = AgenticTrace(
        question_id=question["id"],
        question=question["question"],
        original_id=question.get("original_id", ""),
        answers=question["answers"],
    )

    # Build initial prompt
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT.format(question=question["question"])}
    ]

    for turn in range(max_turns):
        # Call LLM with retry on connection errors
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
            "stop": ["</search>"],
        }

        data = None
        for attempt in range(5):
            try:
                async with session.post(
                    f"{vllm_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        wait = 2 ** attempt + random.uniform(0.5, 1.5)
                        logger.warning(f"HTTP {resp.status} (attempt {attempt+1}), retrying in {wait:.1f}s")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        error_body = await resp.text()
                        trace.status = "error"
                        trace.turns.append({"turn": turn, "error": f"HTTP {resp.status}: {error_body[:300]}"})
                        return trace
                    data = await resp.json()
                    break
            except (asyncio.TimeoutError, aiohttp.ClientError, aiohttp.ServerDisconnectedError) as e:
                wait = 2 ** attempt + random.uniform(0.5, 1.5)
                logger.warning(f"{type(e).__name__} (attempt {attempt+1}/5), retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
            except Exception as e:
                trace.status = "error"
                trace.turns.append({"turn": turn, "error": str(e)})
                return trace

        if data is None:
            trace.status = "error"
            trace.turns.append({"turn": turn, "error": "Failed after 5 retries"})
            return trace

        response_text = data["choices"][0]["message"]["content"]
        finish_reason = data["choices"][0].get("finish_reason", "")

        # Check if model emitted a search query (stopped on </search>)
        if finish_reason == "stop" and "</search>" not in response_text:
            # Model stopped naturally — check for <search> without closing tag
            # (vLLM stop strips the stop token)
            search_query = parse_search_query(response_text + "</search>")
            if search_query is None:
                # Check for answer
                answer = parse_answer(response_text)
                if answer:
                    trace.final_answer = answer
                    trace.status = "answered"
                    trace.turns.append({
                        "turn": turn,
                        "response": response_text,
                        "action": "answer",
                        "answer": answer,
                    })
                    return trace
                # No search, no answer — model just responded
                trace.final_answer = response_text.strip()
                trace.status = "answered"
                trace.turns.append({
                    "turn": turn,
                    "response": response_text,
                    "action": "direct_answer",
                })
                return trace
            # Has search query from partial match
            full_response = response_text + "</search>"
        elif finish_reason == "stop":
            # Check for answer tag first
            answer = parse_answer(response_text)
            if answer:
                trace.final_answer = answer
                trace.status = "answered"
                trace.turns.append({
                    "turn": turn,
                    "response": response_text,
                    "action": "answer",
                    "answer": answer,
                })
                return trace
            search_query = parse_search_query(response_text)
            if search_query is None:
                trace.final_answer = response_text.strip()
                trace.status = "answered"
                trace.turns.append({
                    "turn": turn,
                    "response": response_text,
                    "action": "direct_answer",
                })
                return trace
            full_response = response_text
        else:
            # Stopped on </search> stop sequence
            full_response = response_text + "</search>"
            search_query = parse_search_query(full_response)
            if search_query is None:
                # Fallback: treat the whole response as a search query
                search_query = response_text.strip()
            # Also check if there's an answer before the search
            answer = parse_answer(response_text)
            if answer:
                trace.final_answer = answer
                trace.status = "answered"
                trace.turns.append({
                    "turn": turn,
                    "response": full_response,
                    "action": "answer",
                    "answer": answer,
                })
                return trace

        # Execute search
        trace.num_searches += 1
        query_emb = encoder.encode([search_query])
        distances, indices = faiss_index.search(query_emb, topk)

        retrieved_docs = []
        for idx in indices[0]:
            if 0 <= idx < len(corpus):
                content = corpus[idx].get("contents", corpus[idx].get("content", ""))
                retrieved_docs.append(content)

        info_block = format_search_results(retrieved_docs, topk)

        # Log turn
        trace.turns.append({
            "turn": turn,
            "response": full_response,
            "action": "search",
            "search_query": search_query,
            "query_vector": query_emb[0].tolist(),
            "retrieved_ids": indices[0].tolist(),
            "retrieved_scores": distances[0].tolist(),
        })

        # Append to conversation
        messages.append({"role": "assistant", "content": full_response})
        messages.append({"role": "user", "content": info_block})

    # Exhausted max turns
    trace.status = "max_turns"
    return trace


# ═══════════════════════════════════════════════════════════════
# Batch Evaluation
# ═══════════════════════════════════════════════════════════════

async def evaluate_batch(
    vllm_url: str,
    model_id: str,
    questions: List[dict],
    faiss_index,
    corpus: List[dict],
    encoder: "E5Encoder",
    output_dir: Path,
    index_name: str,
    max_concurrent: int = 8,
    topk: int = 5,
) -> List[AgenticTrace]:
    """Run agentic evaluation on a batch of questions with concurrency control."""

    cache_path = output_dir / "traces" / f"{index_name}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load cache, backfill original_id, skip errors so they get retried
    q_by_id = {q["id"]: q for q in questions}
    cached = {}
    skipped_errors = 0
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                qid = entry["question_id"]
                # Skip errors — they'll be retried
                if entry.get("status") == "error":
                    skipped_errors += 1
                    continue
                # Backfill original_id for old traces
                if "original_id" not in entry and qid in q_by_id:
                    entry["original_id"] = q_by_id[qid].get("original_id", "")
                cached[qid] = entry
        if skipped_errors:
            logger.info(f"  Skipped {skipped_errors} errored traces for retry")
        logger.info(f"  Loaded {len(cached)} cached traces for {index_name}")

    to_eval = [q for q in questions if q["id"] not in cached]
    logger.info(f"  {index_name}: {len(to_eval)} to evaluate, {len(cached)} cached")

    if not to_eval:
        return list(cached.values())

    semaphore = asyncio.Semaphore(max_concurrent)
    results = [AgenticTrace(**v) if isinstance(v, dict) else v for v in cached.values()]
    connector = aiohttp.TCPConnector(limit=max_concurrent)

    async def bounded_episode(session, q):
        async with semaphore:
            return await run_agentic_episode(
                session, vllm_url, model_id, q,
                faiss_index, corpus, encoder, topk=topk,
            )

    batch_size = max_concurrent * 2
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(to_eval), batch_size):
            batch = to_eval[i : i + batch_size]
            tasks = [bounded_episode(session, q) for q in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in batch_results:
                if isinstance(r, BaseException):
                    logger.error(f"  Episode exception: {r}")
                    continue
                results.append(r)

            # Checkpoint
            with open(cache_path, "w") as f:
                for r in results:
                    d = r.to_dict() if isinstance(r, AgenticTrace) else r
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

            done = min(i + batch_size, len(to_eval))
            answered = sum(1 for r in results
                          if (r.status if isinstance(r, AgenticTrace) else r.get("status")) == "answered")
            logger.info(f"  {index_name}: {done}/{len(to_eval)} done, {answered} answered")

    return results


# ═══════════════════════════════════════════════════════════════
# Answer Evaluation (EM / F1)
# ═══════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    """Normalize answer for EM/F1 comparison."""
    import string
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, ground_truths: List[str]) -> bool:
    norm_pred = normalize_answer(prediction)
    return any(normalize_answer(gt) == norm_pred for gt in ground_truths)


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    norm_pred = normalize_answer(prediction)
    pred_tokens = set(norm_pred.split())
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = set(normalize_answer(gt).split())
        common = pred_tokens & gt_tokens
        if not common:
            continue
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        best_f1 = max(best_f1, f1)
    return best_f1


# ═══════════════════════════════════════════════════════════════
# Judge (reuse from naive RAG)
# ═══════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = (
    "You are an expert answer evaluator. Given a question, reference answers, "
    "and a predicted answer, determine whether the prediction is correct.\n\n"
    "Rules:\n"
    '- "correct": The prediction conveys the same core meaning as any reference answer.\n'
    '- "incorrect": The prediction is wrong or irrelevant.\n'
    '- "insufficient": The model stated it cannot answer.\n\n'
    'Output exactly: {"verdict": "correct"|"incorrect"|"insufficient", "explanation": "<brief>"}'
)

JUDGE_USER_TEMPLATE = (
    "Question: {QUERY}\n\n"
    "Reference Answers: {GT_ANSWERS}\n\n"
    "Predicted Answer: {PREDICTION}"
)


async def judge_traces(
    traces: List[dict],
    judge_name: str,
    judge_model_id: str,
    api_key: str,
    output_dir: Path,
    index_name: str,
    max_concurrent: int = 10,
):
    """Judge agentic traces with an LLM judge."""
    cache_path = output_dir / "judgments" / f"{index_name}__{judge_name}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    cached[entry["id"]] = entry

    answerable = [t for t in traces if t.get("final_answer")]
    to_judge = [t for t in answerable if t["question_id"] not in cached]
    logger.info(f"    {judge_name}: {len(cached)} cached, {len(to_judge)} to judge")

    if not to_judge:
        return list(cached.values())

    results = list(cached.values())
    semaphore = asyncio.Semaphore(max_concurrent)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def judge_one(session, trace):
        prompt = JUDGE_USER_TEMPLATE.format(
            QUERY=trace["question"],
            GT_ANSWERS=" | ".join(trace["answers"]),
            PREDICTION=trace["final_answer"],
        )
        payload = {
            "model": judge_model_id,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
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
                            verdict = parsed.get("verdict", "unknown")
                        except json.JSONDecodeError:
                            verdict = "correct" if "correct" in raw.lower() and "incorrect" not in raw.lower() else "incorrect"
                        return {"id": trace["question_id"], "verdict": verdict}
                except Exception:
                    await asyncio.sleep(2 ** attempt)
            return {"id": trace["question_id"], "verdict": "error"}

    batch_size = max_concurrent * 4
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(to_judge), batch_size):
            batch = to_judge[i : i + batch_size]
            tasks = [judge_one(session, t) for t in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in batch_results:
                if isinstance(r, BaseException):
                    logger.error(f"    Judge exception: {r}")
                    continue
                results.append(r)

            # Checkpoint after each batch
            with open(cache_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            done = min(i + batch_size, len(to_judge))
            correct = sum(1 for r in results if r.get("verdict") == "correct")
            logger.info(f"    {judge_name}: {done}/{len(to_judge)} judged, {correct} correct")

    return results


# ═══════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════

def generate_report(all_results: dict, output_dir: Path):
    lines = ["", "=" * 72, "  AGENTIC RAG EVALUATION REPORT", "=" * 72]

    for index_name, data in sorted(all_results.items()):
        traces = data["traces"]
        answered = [t for t in traces if (t.get("status") or t.status) == "answered"]
        total = len(traces)

        # EM / F1
        em_scores = []
        f1_scores = []
        for t in answered:
            ans = t.get("final_answer") or t.final_answer
            gts = t.get("answers") or t.answers
            if ans and gts:
                em_scores.append(1.0 if exact_match(ans, gts) else 0.0)
                f1_scores.append(f1_score(ans, gts))

        avg_searches = np.mean([
            t.get("num_searches", 0) if isinstance(t, dict) else t.num_searches
            for t in traces
        ])

        lines.append(f"\n{'─' * 60}")
        lines.append(f"  Index: {index_name}")
        lines.append(f"  Answered: {len(answered)}/{total}  Avg searches: {avg_searches:.1f}")
        if em_scores:
            lines.append(f"  EM: {np.mean(em_scores):.3f}  F1: {np.mean(f1_scores):.3f}")

        # Judge results if available
        for judge_name, judgments in data.get("judgments", {}).items():
            correct = sum(1 for j in judgments if j.get("verdict") == "correct")
            lines.append(f"  {judge_name}: {correct}/{len(judgments)} correct ({correct/len(judgments)*100:.1f}%)")

    report = "\n".join(lines)
    print(report)
    (output_dir / "report.txt").write_text(report)


# ═══════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════

def cmd_prepare_index(args):
    """Build HNSW and IVF indices from flat index."""
    # Extract embeddings
    extract_embeddings_from_flat(FLAT_INDEX_PATH, EMBEDDING_PATH)
    embeddings = load_embedding(EMBEDDING_PATH)

    # Build HNSW
    hnsw_path = HNSW_DIR / f"e5_HNSW{HNSW_CONFIGS['M']}.index"
    build_hnsw_index(embeddings, HNSW_CONFIGS["M"], HNSW_CONFIGS["efConstruction"], hnsw_path)

    # Build IVF
    ivf_path = IVF_DIR / f"e5_IVF{IVF_CONFIGS['n_list']}.index"
    build_ivf_index(embeddings, IVF_CONFIGS["n_list"], ivf_path)

    # Compute GT if needed
    if not GT_PATH.exists():
        logger.info("Computing ground truth with flat index...")
        flat = faiss.read_index(str(FLAT_INDEX_PATH))
        # We'll compute GT later when we have query vectors
        logger.info("GT will be computed during first run (needs query vectors from model)")

    logger.info("Index preparation complete.")


async def cmd_run(args):
    """Run agentic evaluation."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    questions = load_hotpotqa(HOTPOTQA_DATA, max_questions=args.max_questions)

    # Filter to test set if requested
    if args.test_only:
        test_ids_path = Path(args.output_dir) / "test_ids.json"
        if not test_ids_path.exists():
            logger.error(f"test_ids.json not found at {test_ids_path}. Run tuning split first.")
            return
        test_ids = set(json.load(open(test_ids_path)))
        questions = [q for q in questions if q["id"] in test_ids]
        logger.info(f"Filtered to {len(questions)} test set questions")

    corpus = load_corpus(CORPUS_PATH)

    # Load encoder
    logger.info("Loading E5 encoder...")
    encoder = E5Encoder(E5_MODEL, device=args.encoder_device)

    # Prepare index configs to evaluate
    eval_configs = []

    # KNN baseline (flat index, GPU-accelerated)
    logger.info("Loading flat index...")
    flat = faiss.read_index(str(FLAT_INDEX_PATH))
    if args.knn_gpu >= 0:
        logger.info(f"Moving flat index to GPU {args.knn_gpu}...")
        res = faiss.StandardGpuResources()
        flat = faiss.index_cpu_to_gpu(res, args.knn_gpu, flat)
    eval_configs.append(("knn_flat", flat, {}))

    if not args.knn_only:
        # HNSW configs (hnswlib format)
        hnsw_path = HNSW_DIR / f"e5_HNSW{HNSW_CONFIGS['M']}.hnswlib"
        if hnsw_path.exists():
            import hnswlib
            for ef in HNSW_CONFIGS["efSearch_values"]:
                hnsw = HnswlibWrapper(str(hnsw_path), dim=768, ef=ef)
                eval_configs.append((f"hnsw_ef{ef}", hnsw, {"efSearch": ef}))
        else:
            logger.warning(f"HNSW index not found: {hnsw_path}. Run build_hnsw.py first.")

        # IVF configs
        ivf_path = IVF_DIR / f"e5_IVF{IVF_CONFIGS['n_list']}.index"
        if ivf_path.exists():
            for np_ in IVF_CONFIGS["n_probe_values"]:
                ivf = faiss.read_index(str(ivf_path))
                ivf.nprobe = np_
                eval_configs.append((f"ivf_np{np_}", ivf, {"nprobe": np_}))
        else:
            logger.warning(f"IVF index not found: {ivf_path}. Run 'prepare-index' first.")

    all_results = {}
    for index_name, faiss_index, params in eval_configs:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Evaluating: {index_name} (params={params})")
        logger.info(f"{'=' * 50}")

        traces = await evaluate_batch(
            args.vllm_url, VLLM_MODEL, questions,
            faiss_index, corpus, encoder, output_dir, index_name,
            max_concurrent=args.max_concurrent,
            topk=args.topk,
        )

        trace_dicts = [t.to_dict() if isinstance(t, AgenticTrace) else t for t in traces]
        all_results[index_name] = {"traces": trace_dicts}

    # Quick EM/F1 report
    generate_report(all_results, output_dir)


async def cmd_judge(args):
    """Judge previously generated traces."""
    output_dir = Path(args.output_dir)
    traces_dir = output_dir / "traces"

    if not traces_dir.exists():
        logger.error("No traces found. Run 'run' first.")
        return

    all_results = {}
    for trace_file in sorted(traces_dir.glob("*.jsonl")):
        index_name = trace_file.stem
        traces = []
        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))

        logger.info(f"Judging {index_name}: {len(traces)} traces")
        judgments = {}
        for judge_name, judge_id in JUDGE_MODELS.items():
            j = await judge_traces(
                traces, judge_name, judge_id, args.openrouter_key,
                output_dir, index_name,
            )
            judgments[judge_name] = j

        all_results[index_name] = {"traces": traces, "judgments": judgments}

    generate_report(all_results, output_dir)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Agentic RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare-index
    sub = subparsers.add_parser("prepare-index", help="Build HNSW/IVF indices")

    # run
    sub = subparsers.add_parser("run", help="Run agentic evaluation")
    sub.add_argument("--vllm-url", default=VLLM_DEFAULT_URL)
    sub.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    sub.add_argument("--max-questions", type=int, default=2000)
    sub.add_argument("--max-concurrent", type=int, default=8)
    sub.add_argument("--topk", type=int, default=5)
    sub.add_argument("--encoder-device", default="cuda:0")
    sub.add_argument("--knn-only", action="store_true", help="Only run KNN baseline (collect traces)")
    sub.add_argument("--knn-gpu", type=int, default=-1, help="GPU ID for flat index search (-1 for CPU)")
    sub.add_argument("--test-only", action="store_true", help="Only run test set questions (from test_ids.json)")

    # judge
    sub = subparsers.add_parser("judge", help="Judge traces with LLM judges")
    sub.add_argument("--openrouter-key", default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    sub.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    args = parser.parse_args()

    if args.command == "prepare-index":
        cmd_prepare_index(args)
    elif args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "judge":
        asyncio.run(cmd_judge(args))


if __name__ == "__main__":
    main()
