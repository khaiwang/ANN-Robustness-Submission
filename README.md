# ANN Robustness Benchmark

Artifact for *"Towards Robustness: A Critique of Current Vector Database Assessments"* (VLDB 2026).

We extend [Big-ANN-Benchmarks](http://big-ann-benchmarks.com/) with the Robustness-delta@K metric and evaluate 6 vector indexes on 4 datasets.

## Quick Start: Reproduce Paper Figures

```bash
pip install -r requirements.txt
cd scripts && python generate_all_figures.py
```

Figures are written to `output/`. No datasets, Docker, or HDF5 files needed.

## Benchmark Setup

### Indexes
HNSW (Faiss), DiskANN, Zilliz, IVFFlat (Faiss), ScaNN, Puck

### Datasets
Text-to-Image-10M, MSSPACEV-10M, DEEP-10M (from Big-ANN-Benchmarks NeurIPS'23 OOD track), and MSMARCO (8.8M passages, encoded with LLM-Embedder, 768-dim inner product).

Zilliz is excluded from MSMARCO due to a bug in its Docker image quantizing 768-dim vectors.

### RAG Applications
- Naive RAG Q&A: MSMARCO + Gemini-2.0-Flash, 4 indexes, K=10
- Agentic RAG: HotpotQA + Search-R1 (Qwen2.5-7B and Qwen3-30B-A3B), HNSW vs IVF, K=5

## Data

Pre-computed CSV data from benchmark results. See `data/README_DATA.md` for schemas.

| Directory | Content |
|---|---|
| `data/aggregate/` | Per-configuration metrics: recall, robustness at 5 delta thresholds, QPS |
| `data/cdf/` | 11-point robustness CDF at recall~0.9 (one config per algorithm per dataset) |
| `data/metric_comparison/` | Extended metrics (MAP, NDCG, MRR, percentiles) for Section 4 analysis |
| `data/rag/` | End-to-end RAG accuracy for naive and agentic setups |

## Figure Map

| Paper Figure | Section | Script | Data |
|---|---|---|---|
| Fig 1 | 1 | `fig01_recall_distribution.py` | `cdf/msmarco_10M_k10_cdf.csv` |
| Fig 2 | 2 | (static illustration) | -- |
| Fig 3 | 4 | `fig03_metric_correlation.py` | `metric_comparison/all_datasets_all_metrics.csv` |
| Fig 4--5 | 5 | (LaTeX tables) | -- |
| Fig 6 | 5.1 | `fig06_cdf_split.py` | `cdf/text2image_10M_k{10,100}_cdf.csv` |
| Fig 7 | 5.1 | `fig07_cdf_stacked.py` | `cdf/{msspacev,deep,msmarco}_10M_k10_cdf.csv` |
| Fig 8 | 5.1 | `fig08_recall_robustness.py` | `aggregate/text2image_10M_k10.csv` |
| Fig 9 | 5.2 | `fig09_tradeoff.py` | `aggregate/text2image_10M_k10.csv` |
| Fig 10--11 | 5.3 | (LaTeX tables) | -- |
| Fig 12 | 5.3 | `fig12_rag.py` | `rag/rag_results.csv` |
| Fig 13--14 | 5.4 | (pre-generated from Eval 1 results) | -- |

## Running the Full Benchmark from Scratch

See `README_SUBMISSION.md` for step-by-step instructions to reproduce all experiments from raw datasets. This requires Docker, the Big-ANN-Benchmarks datasets, and API keys for RAG evaluation.

The `extract/` directory contains the scripts used to produce CSV data from raw HDF5 results. See `extract/README_EXTRACT.md`.

## Credits

This project extends [Big-ANN-Benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks) (NeurIPS'23 OOD track).
