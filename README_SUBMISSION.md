# Reproducing All Experiments from Scratch

This document describes how to reproduce all benchmark results from raw datasets.
For generating paper figures from pre-computed CSV data, see `README.md`.

## Prerequisites

- Python 3.10+ virtual environment
- Docker 27+
- Datasets: Text-to-Image-10M, MSSPACEV-10M, DEEP-10M (from Big-ANN-Benchmarks), MSMARCO (encoded with LLM-Embedder)
- For RAG: API keys for OpenRouter (Gemini-2.0-Flash, GPT-4o-mini judges), vLLM server for Search-R1/Qwen3-30B

```bash
python3 -m venv bigann && source bigann/bin/activate
pip install -r requirements.txt        # minimal deps for plotting and extraction
pip install -r requirements_py3.10.txt  # full deps if running benchmarks locally (Python 3.10)
```

Note: The benchmark indexes run inside Docker containers with their own environments. The host machine only needs `requirements.txt` for metric extraction and figure generation. `requirements_py3.10.txt` contains pinned versions for the full Big-ANN-Benchmarks framework and may require Python 3.10.

## Eval 1: Index Performance (Section 5.1, Figs 6--8)

Evaluates 6 indexes on 4 datasets with K=10. Produces CDF robustness curves and recall-robustness scatter plots.

**Indexes**: HNSW, DiskANN, Zilliz, IVFFlat, ScaNN, Puck
**Datasets**: Text-to-Image-10M, MSSPACEV-10M, DEEP-10M, MSMARCO
**Note**: Zilliz is excluded from MSMARCO (Docker image bug quantizing 768-dim vectors).

```bash
# 1. Install index Docker images
python robustness_evaluation.py --run install

# 2. Run benchmarks (results saved to results/neurips23/ood/)
python robustness_evaluation.py --run run --count 10

# 3. Extract metrics to CSV
#    extract_all_metrics.py outputs to data/OverallEval/
#    extract/ scripts output to their sibling data/ directory (using __file__-relative paths)
#    These must be run from the benchmark root with access to results/ and datasets
PYTHONPATH="." python extract_all_metrics.py

# 4. Generate figures
cd scripts && python fig06_cdf_split.py && python fig07_cdf_stacked.py && python fig08_recall_robustness.py
```

Index configurations are in `neurips23/ood/<index_name>/config.yaml`.
Index implementations are in `neurips23/ood/<index_name>/<index_name>.py`.

**K=100 evaluation** (for Fig 6, bottom row): Set `--count 100` when running. Note that ScaNN and Puck depend on `ds.default_count()`, so you must manually set the return value to 100 in `benchmark/datasets.py` for the `Text2Image1B` and `Dataset` classes. For ScaNN, also set `num_neighbors=100` in `neurips23/ood/scann/scann.py`.

## Eval 2: Three-Way Tradeoff (Section 5.2, Fig 9)

Shows recall-robustness-throughput tradeoff for Zilliz and ScaNN on Text-to-Image-10M.
Uses results from Eval 1 (no additional benchmark runs needed).

```bash
cd scripts && python fig09_tradeoff.py
```

## Eval 3: RAG Applications (Section 5.3, Fig 12)

Two RAG applications demonstrating that robustness predicts end-to-end accuracy.

### Naive RAG Q&A (MSMARCO)
- LLM: Gemini-2.0-Flash (via OpenRouter)
- Embedder: LLM-Embedder (768-dim, inner product)
- Indexes: HNSW, IVFFlat, ScaNN, DiskANN (K=10)
- Judge: GPT-4o-mini (via OpenRouter), verified with Gemini-2.0-Flash

```bash
cd rag
python naive_rag_pipeline.py --openrouter-key <KEY>
```

### Agentic RAG (HotpotQA)
- Models: Search-R1 (Qwen2.5-7B) and Qwen3-30B-A3B (via vLLM)
- Corpus: Wikipedia 18M passages, encoded with E5
- Indexes: HNSW, IVF (K=5)
- Judge: GPT-4o-mini (via OpenRouter)

```bash
cd rag
# Build indices (requires E5 flat index and corpus)
python agentic_rag_pipeline.py prepare-index

# Run evaluation (requires vLLM server)
python agentic_rag_pipeline.py run --vllm-url http://localhost:8001/v1

# Judge results
python agentic_rag_pipeline.py judge --openrouter-key <KEY>
```

### Generate RAG figure
```bash
cd scripts && python fig12_rag.py
```

## Eval 4: Index Family Analysis (Section 5.4, Figs 13--14)

Parameter study for HNSW (M, efSearch) and IVFFlat (n_probe) on Text-to-Image-10M.
Uses results from Eval 1.

```bash
# HNSW parameter study
python plot.py -x k-nn -y qps --dataset text2image-10M --neurips23track ood \
    --count 10 --definitions neurips23/ood/faiss_hnsw/config.yaml

# IVFFlat parameter study
python plot.py -x k-nn -y qps --dataset text2image-10M --neurips23track ood \
    --count 10 --definitions neurips23/ood/faiss/config.yaml
```

Results are plotted manually for the paper (the benchmark framework does not support multi-line parameter sweeps natively). The `analyze_index_families.py` script provides GT rank distribution analysis and K=100 retrieve-and-rerank analysis.

## Eval 5: Metric Comparison (Section 4, Fig 3)

Systematic r^2 correlation analysis between Robustness-delta and average Recall@10, compared with MAP, NDCG, MRR, and percentile metrics across all 4 datasets.

```bash
# Extract all metrics (recall, robustness, MAP, NDCG, MRR, percentiles)
PYTHONPATH="." python extract_all_metrics.py

# Generate correlation heatmap
cd scripts && python fig03_metric_correlation.py
```

