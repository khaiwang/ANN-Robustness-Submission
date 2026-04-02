# Data Schemas

## Aggregate Metrics (`aggregate/*.csv`)

Per-configuration metrics for all evaluated index configurations. One file per dataset/K combination.

| Column | Type | Description |
|---|---|---|
| `algorithm` | string | Display name: HNSW, DiskANN, Zilliz, IVFFlat, ScaNN, Puck |
| `config` | string | Configuration identifier (from HDF5 filename) |
| `recall` | float | Mean Recall@K across all queries |
| `robustness_0.1` ... `robustness_0.9` | float | P(per-query recall >= delta), for delta in {0.1, 0.3, 0.5, 0.7, 0.9} |
| `qps` | float | Queries per second |

Files: `text2image_10M_k10.csv` (120 configs), `msspacev_10M_k10.csv` (103), `deep_10M_k10.csv` (94), `msmarco_10M_k10.csv` (73), `text2image_10M_k100.csv` (103).

Extracted by `extract/extract_aggregate.py` from `results/neurips23/ood/{dataset}/{K}/{algo}/*.hdf5`.

## CDF Data (`cdf/*.csv`)

Robustness CDF (11-point distribution) for the configuration closest to recall=0.9 per algorithm. One row per algorithm.

| Column | Type | Description |
|---|---|---|
| `algorithm` | string | Display name |
| `config` | string | Selected configuration identifier |
| `recall` | float | Actual recall of selected config (target: 0.9, tolerance: 0.02) |
| `cdf_0.0` ... `cdf_1.0` | float | P(per-query recall >= delta * K), for delta = 0.0, 0.1, ..., 1.0 |

CDF computation: `cdf[d] = mean(per_query_recalls >= d * K - 0.5)`, with `cdf[0] = 1.0`. The `-0.5` correction handles floating-point boundaries for integer recall counts.

Extracted by `extract/extract_cdf.py`.

## Metric Comparison (`metric_comparison/all_datasets_all_metrics.csv`)

Extended metrics for the Section 4 correlation analysis. Includes all index configurations across all 4 datasets.

| Column | Type | Description |
|---|---|---|
| `dataset` | string | Dataset name |
| `algorithm` | string | Algorithm directory name (e.g., `faiss_hnsw`, `scann`) |
| `config` | string | Full parameter string |
| `recall@10` | float | Mean Recall@10 |
| `qps` | float | Queries per second |
| `robustness@0.1` ... `robustness@0.9` | float | Robustness at 5 delta values |
| `map@10` | float | Mean Average Precision@10 |
| `ndcg@10` | float | Normalized Discounted Cumulative Gain@10 |
| `mrr@10` | float | Mean Reciprocal Rank@10 |
| `tail95`, `tail99`, `tail999` | float | Recall percentile values |

Extracted by `extract_all_metrics.py` (in ANN-Robustness-Revision).

## RAG Results (`rag/rag_results.csv`)

End-to-end RAG evaluation accuracy for naive and agentic RAG applications.

| Column | Type | Description |
|---|---|---|
| `scenario` | string | `naive`, `agentic_searchr1`, or `agentic_qwen3` |
| `algorithm` | string | HNSW, IVFFlat, DiskANN, ScaNN, or K-NN |
| `k` | int | Top-K for retrieval (10 for naive, 5 for agentic) |
| `delta` | float | Robustness threshold (0.2 for naive, 0.4 for agentic) |
| `recall` | float | Average Recall@K |
| `robustness` | float | Robustness at the specified delta |
| `accuracy` | float | End-to-end application accuracy |
