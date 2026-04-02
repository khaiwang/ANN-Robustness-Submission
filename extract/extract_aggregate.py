#!/usr/bin/env python3
"""
Extract aggregate metrics from HDF5 result files into CSV.
Must run from ANN-Robustness-Revision/ with PYTHONPATH=".".

Usage:
    cd /path/to/ANN-Robustness-Revision
    PYTHONPATH="." python ../ANN-Robustness-Submission/extract/extract_aggregate.py
"""
import os
import csv
import h5py
import numpy as np
from benchmark.datasets import DATASETS
from benchmark.plotting.metrics import get_recall_values

RESULTS_DIR = "results/neurips23/ood"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "aggregate")

ALGO_MAP = {
    'faiss_hnsw': 'HNSW',
    'diskann': 'DiskANN',
    'zilliz': 'Zilliz',
    'faiss-ivf': 'IVFFlat',
    'scann': 'ScaNN',
    'puck': 'Puck',
}

DATASETS_K = [
    # (dataset_name, K_values)
    ("text2image-10M", [10, 100]),
    ("msspacev-10M", [10]),
    ("deep-10M", [10]),
    ("msmarco-10M", [10]),
]

ROBUSTNESS_DELTAS = [0.1, 0.3, 0.5, 0.7, 0.9]

HEADER = ["algorithm", "config", "recall",
          "robustness_0.1", "robustness_0.3", "robustness_0.5",
          "robustness_0.7", "robustness_0.9", "qps"]


def extract_dataset(dataset_name, count):
    ds = DATASETS[dataset_name]()
    gt = ds.get_groundtruth(count)

    ds_dir = os.path.join(RESULTS_DIR, dataset_name, str(count))
    if not os.path.exists(ds_dir):
        print(f"  Skipping {dataset_name}/K={count}: no results dir")
        return []

    rows = []
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
                    search_times = np.array(f.attrs.get(
                        'search_times',
                        [f.attrs.get('best_search_time', 1.0)]
                    ))
                    best_time = float(np.min(search_times))
                    n_queries = neighbors.shape[0]
                    qps = n_queries / best_time if best_time > 0 else 0

                    recall_mean, _, recalls, _ = get_recall_values(gt, neighbors, count)
                    recalls_norm = recalls / float(count)

                    robustness = {d: float(np.mean(recalls_norm >= d))
                                  for d in ROBUSTNESS_DELTAS}

                    config_name = fname.replace('.hdf5', '')
                    rows.append([
                        display_name, config_name, f"{recall_mean:.6f}",
                        *[f"{robustness[d]:.6f}" for d in ROBUSTNESS_DELTAS],
                        f"{qps:.2f}",
                    ])
            except Exception as e:
                print(f"  Error: {fpath}: {e}")

    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 0

    for dataset_name, k_values in DATASETS_K:
        for k in k_values:
            print(f"Extracting {dataset_name} K={k}...")
            rows = extract_dataset(dataset_name, k)
            if not rows:
                continue

            safe_name = dataset_name.replace('-', '_')
            out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_k{k}.csv")
            with open(out_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(HEADER)
                w.writerows(rows)

            print(f"  Wrote {len(rows)} rows to {out_path}")
            total += len(rows)

    print(f"\nTotal: {total} configurations extracted")


if __name__ == "__main__":
    main()
