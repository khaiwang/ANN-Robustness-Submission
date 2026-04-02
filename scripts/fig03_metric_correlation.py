#!/usr/bin/env python3
"""Fig 3 (Section 4): Metric correlation r^2 heatmap and family comparison.
Adapted from plot_metric_comparison.py (already CSV-based)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import stats
from common import DATA_DIR, OUTPUT_DIR

CSV_PATH = DATA_DIR / "metric_comparison" / "all_datasets_all_metrics.csv"
DATASETS = ['text2image-10M', 'msspacev-10M', 'deep-10M', 'msmarco-10M']
DATASET_LABELS = ['Text-to-Image', 'MSSPACEV', 'DEEP', 'MSMARCO']
RECALL_MIN, RECALL_MAX = 0.70, 0.95

METRICS = [
    ('map@10', 'MAP@10'),
    ('ndcg@10', 'NDCG@10'),
    ('mrr@10', 'MRR@10'),
    ('tail95', '95%ile'),
    ('tail99', '99%ile'),
    ('robustness@0.1', 'Rob-0.1'),
    ('robustness@0.3', 'Rob-0.3'),
    ('robustness@0.9', 'Rob-0.9'),
]


def load_data():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r['dataset'] == 'msmarco-10M' and 'ivfpqfs' in r.get('algorithm', ''):
                continue
            recall = float(r['recall@10'])
            if RECALL_MIN <= recall <= RECALL_MAX:
                rows.append(r)
    return rows


def plot_r2_heatmap(rows):
    n_metrics = len(METRICS)
    n_datasets = len(DATASETS)
    r2_matrix = np.zeros((n_metrics, n_datasets))

    for j, ds in enumerate(DATASETS):
        ds_rows = [r for r in rows if r['dataset'] == ds]
        recalls = np.array([float(r['recall@10']) for r in ds_rows])
        for i, (col, _) in enumerate(METRICS):
            vals = np.array([float(r[col]) for r in ds_rows])
            mask = ~np.isnan(vals) & ~np.isnan(recalls)
            if mask.sum() > 2:
                slope, intercept, r_val, p, se = stats.linregress(recalls[mask], vals[mask])
                r2_matrix[i, j] = r_val ** 2
            else:
                r2_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(5, 4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('gr', ['#228833', '#FFFFFF', '#EE6677'])
    im = ax.imshow(r2_matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(n_datasets))
    ax.set_xticklabels(DATASET_LABELS, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels([label for _, label in METRICS], fontsize=9)

    for i in range(n_metrics):
        for j in range(n_datasets):
            val = r2_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if val > 0.7 or val < 0.3 else 'black')

    plt.colorbar(im, ax=ax, label='$r^2$', shrink=0.8)
    ax.set_title('Correlation with Average Recall@10', fontsize=10)

    out = OUTPUT_DIR / 'fig03_metric_correlation_r2.pdf'
    plt.tight_layout()
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = load_data()
    print(f"Loaded {len(rows)} configurations from CSV")
    plot_r2_heatmap(rows)


if __name__ == '__main__':
    main()
