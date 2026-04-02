"""
Extract all metrics (recall, robustness, MAP, NDCG, MRR, percentiles, QPS)
from HDF5 result files into CSV for the metric comparison analysis (§4 revision).
"""
import os
import sys
import csv
import h5py
import numpy as np
from benchmark.datasets import DATASETS
from benchmark.results import load_all_results
from benchmark.plotting.metrics import (
    knn, robustness, IR, tail_knn, get_recall_values, get_IR_values
)

TRACK = "ood"
COUNT = 10
DATASETS_LIST = ["text2image-10M", "msspacev-10M", "deep-10M", "msmarco-10M"]
ROBUSTNESS_DELTAS = [0.1, 0.3, 0.5, 0.7, 0.9]

def extract_metrics_for_dataset(dataset_name):
    """Extract all metrics for every configuration in a dataset."""
    print(f"\n=== Processing {dataset_name} ===")

    ds = DATASETS[dataset_name]()
    # Load ground truth
    true_nn = ds.get_groundtruth(COUNT)

    # Load all results
    results_dir = os.path.join("results", "neurips23", TRACK, dataset_name, str(COUNT))
    if not os.path.exists(results_dir):
        print(f"  No results directory: {results_dir}")
        return []

    # Collect all HDF5 files recursively (nested in per-algo subdirs)
    hdf5_files = []
    for root, dirs, files in os.walk(results_dir):
        for fn in sorted(files):
            if fn.endswith(".hdf5"):
                hdf5_files.append(os.path.join(root, fn))
    print(f"  Found {len(hdf5_files)} result files")

    rows = []
    for filepath in hdf5_files:
        fn = os.path.basename(filepath)
        algo_dir = os.path.basename(os.path.dirname(filepath))
        try:
            with h5py.File(filepath, "a") as f:
                # Get algorithm info
                algo = f.attrs.get("algo", algo_dir)
                name = f.attrs.get("name", fn.replace(".hdf5", ""))

                run_nn = np.array(f["neighbors"])
                count = int(f.attrs.get("count", COUNT))

                # Get or create metrics cache
                if "metrics" not in f:
                    metrics = f.create_group("metrics")
                else:
                    metrics = f["metrics"]

                # 1. Recall
                knn_result = knn(true_nn, run_nn, count, metrics)
                recall_mean = float(knn_result.attrs["mean"])

                # 2. Robustness at various deltas
                rob_values = {}
                for delta in ROBUSTNESS_DELTAS:
                    val = robustness(true_nn, run_nn, count, metrics, delta)
                    rob_values[delta] = float(val)

                # 3. IR metrics (MAP, NDCG, MRR)
                ir_values = {}
                for ir_name in ["map", "ndcg", "mrr"]:
                    val = IR(true_nn, run_nn, count, metrics, ir_name)
                    ir_values[ir_name] = float(val)

                # 4. Percentile metrics
                tail_values = {}
                for tp in [95, 99, 999]:
                    val = tail_knn(true_nn, run_nn, count, metrics, tp)
                    tail_values[tp] = float(val)

                # 5. QPS
                qps = float(f.attrs.get("best_search_time", 0))
                if qps > 0:
                    n_queries = run_nn.shape[0]
                    qps = n_queries / qps
                else:
                    qps = 0

                row = {
                    "dataset": dataset_name,
                    "algorithm": str(algo),
                    "config": str(name),
                    "recall@10": recall_mean,
                    "qps": qps,
                }
                for delta in ROBUSTNESS_DELTAS:
                    row[f"robustness@{delta}"] = rob_values[delta]
                for ir_name in ["map", "ndcg", "mrr"]:
                    row[f"{ir_name}@10"] = ir_values[ir_name]
                for tp in [95, 99, 999]:
                    row[f"tail{tp}"] = tail_values[tp]

                rows.append(row)
                print(f"  {name}: recall={recall_mean:.4f}, map={ir_values['map']:.4f}, "
                      f"ndcg={ir_values['ndcg']:.4f}, mrr={ir_values['mrr']:.4f}, "
                      f"rob@0.3={rob_values[0.3]:.4f}")
        except Exception as e:
            print(f"  ERROR processing {fn}: {e}")
            import traceback
            traceback.print_exc()

    return rows


def main():
    os.makedirs("data/OverallEval", exist_ok=True)
    all_rows = []

    for dataset_name in DATASETS_LIST:
        rows = extract_metrics_for_dataset(dataset_name)
        all_rows.extend(rows)

        # Write per-dataset CSV
        if rows:
            outfile = f"data/OverallEval/{dataset_name.replace('-', '_')}_all_metrics.csv"
            fieldnames = rows[0].keys()
            with open(outfile, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Written {len(rows)} rows to {outfile}")

    # Write combined CSV
    if all_rows:
        outfile = "data/OverallEval/all_datasets_all_metrics.csv"
        fieldnames = all_rows[0].keys()
        with open(outfile, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n=== Written {len(all_rows)} total rows to {outfile} ===")


if __name__ == "__main__":
    main()
