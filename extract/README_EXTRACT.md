# Data Extraction Scripts

These scripts extract CSV data from HDF5 benchmark results. They are provided for transparency and reproducibility, but are **not needed** to generate figures (the CSV data is already included in `data/`).

## Requirements

- The full ANN-Robustness benchmark codebase (`ANN-Robustness-Revision/`)
- Benchmark datasets (Text-to-Image-10M, MSSPACEV-10M, DEEP-10M, MSMARCO) with ground truth
- HDF5 result files in `results/neurips23/ood/`
- Python packages: `h5py`, `numpy`, plus the `benchmark` module from the codebase

## Usage

Run from the benchmark codebase root:

```bash
cd /path/to/ANN-Robustness-Revision

# Extract aggregate metrics (recall, robustness, QPS per config)
PYTHONPATH="." python /path/to/ANN-Robustness-Submission/extract/extract_aggregate.py

# Extract CDF data (11-point distribution at recall~0.9)
PYTHONPATH="." python /path/to/ANN-Robustness-Submission/extract/extract_cdf.py
```

## What they produce

- `extract_aggregate.py` writes to `data/aggregate/*.csv` (one file per dataset/K)
- `extract_cdf.py` writes to `data/cdf/*.csv` (one file per dataset/K, one row per algorithm)
- `data/metric_comparison/all_datasets_all_metrics.csv` was produced by `extract_all_metrics.py` in the benchmark codebase
- `data/rag/rag_results.csv` was manually compiled from RAG experiment outputs
