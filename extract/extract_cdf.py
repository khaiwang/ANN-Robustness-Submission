#!/usr/bin/env python3
"""
Extract CDF data (11-point robustness distribution) from HDF5 results.
For each dataset/K, selects the config closest to recall=0.9 per algorithm.
Must run from ANN-Robustness-Revision/ with PYTHONPATH=".".

Usage:
    cd /path/to/ANN-Robustness-Revision
    PYTHONPATH="." python ../ANN-Robustness-Submission/extract/extract_cdf.py
"""
import os
import csv
import h5py
import numpy as np
from benchmark.datasets import DATASETS
from benchmark.plotting.metrics import get_recall_values

RESULTS_DIR = "results/neurips23/ood"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cdf")

ALGO_MAP = {
    'faiss_hnsw': 'HNSW',
    'diskann': 'DiskANN',
    'zilliz': 'Zilliz',
    'faiss-ivf': 'IVFFlat',
    'scann': 'ScaNN',
    'puck': 'Puck',
}

DATASETS_K = [
    ("text2image-10M", [10, 100]),
    ("msspacev-10M", [10]),
    ("deep-10M", [10]),
    ("msmarco-10M", [10]),
]

CDF_DELTAS = [i / 10.0 for i in range(11)]  # 0.0, 0.1, ..., 1.0
TARGET_RECALL = 0.9
TOLERANCE = 0.02


def load_all_with_cdf(dataset_name, count):
    """Load HDF5 results and compute CDF for each config."""
    ds = DATASETS[dataset_name]()
    gt = ds.get_groundtruth(count)

    ds_dir = os.path.join(RESULTS_DIR, dataset_name, str(count))
    if not os.path.exists(ds_dir):
        return []

    results = []
    for algo_dir in sorted(os.listdir(ds_dir)):
        display_name = ALGO_MAP.get(algo_dir)
        if display_name is None:
            continue

        algo_path = os.path.join(ds_dir, algo_dir)
        for fname in sorted(os.listdir(algo_path)):
            if not fname.endswith('.hdf5'):
                continue
            fpath = os.path.join(algo_path, fname)
            try:
                with h5py.File(fpath, 'r') as f:
                    neighbors = np.array(f['neighbors'])
                    recall_mean, _, recalls, _ = get_recall_values(gt, neighbors, count)

                    # CDF: P(per-query recall >= delta * K)
                    # The -0.5 handles floating-point boundary for integer recalls
                    cdf_values = [float(np.mean(recalls >= d * count - 0.5))
                                  for d in CDF_DELTAS]
                    cdf_values[0] = 1.0  # delta=0: all queries have recall >= 0

                    results.append({
                        'algo': display_name,
                        'config': fname.replace('.hdf5', ''),
                        'recall': float(recall_mean),
                        'cdf_values': cdf_values,
                    })
            except Exception as e:
                print(f"  Error: {fpath}: {e}")

    return results


def find_config_at_recall(results, target, tolerance):
    """Select the config closest to target recall per algorithm."""
    by_algo = {}
    for r in results:
        algo = r['algo']
        if algo not in by_algo or abs(r['recall'] - target) < abs(by_algo[algo]['recall'] - target):
            if abs(r['recall'] - target) <= tolerance:
                by_algo[algo] = r
    return by_algo


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    header = ["algorithm", "config", "recall"] + [f"cdf_{d:.1f}" for d in CDF_DELTAS]

    for dataset_name, k_values in DATASETS_K:
        for k in k_values:
            print(f"Extracting CDF for {dataset_name} K={k}...")
            results = load_all_with_cdf(dataset_name, k)
            if not results:
                print(f"  No results found")
                continue

            selected = find_config_at_recall(results, TARGET_RECALL, TOLERANCE)
            if not selected:
                print(f"  No configs near recall={TARGET_RECALL}")
                continue

            safe_name = dataset_name.replace('-', '_')
            out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_k{k}_cdf.csv")
            with open(out_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(header)
                for algo in ['HNSW', 'DiskANN', 'Zilliz', 'IVFFlat', 'ScaNN', 'Puck']:
                    if algo not in selected:
                        continue
                    r = selected[algo]
                    w.writerow([
                        r['algo'], r['config'], f"{r['recall']:.6f}",
                        *[f"{v:.6f}" for v in r['cdf_values']],
                    ])

            print(f"  Wrote {len(selected)} algos to {out_path}")


if __name__ == "__main__":
    main()
