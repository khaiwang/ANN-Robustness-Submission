"""
Generate figures for §4 metric comparison revision.
Figure 1: r² correlation of each metric with average recall across 3 datasets
Figure 2: At fixed recall (~0.9), how do metrics differ between graph vs partition families?
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load data
with open('data/OverallEval/all_datasets_all_metrics.csv') as f:
    rows = list(csv.DictReader(f))

datasets = ['text2image-10M', 'msspacev-10M', 'deep-10M', 'msmarco-10M']
ds_labels = ['Text-to-Image', 'MSSPACEV', 'DEEP', 'MSMARCO']

# Exclude faiss-ivfpqfs from MSMARCO (user requested)
rows = [r for r in rows if not (r['dataset'] == 'msmarco-10M' and 'ivfpqfs' in r.get('algorithm', '').lower())]

# ============================================================
# Figure 1: r² correlation heatmap
# ============================================================
metrics_for_corr = [
    ('map@10', 'MAP@10'),
    ('ndcg@10', 'NDCG@10'),
    ('mrr@10', 'MRR@10'),
    ('tail95', '95%ile'),
    ('tail99', '99%ile'),
    ('robustness@0.1', 'Rob.@0.1'),
    ('robustness@0.3', 'Rob.@0.3'),
    ('robustness@0.9', 'Rob.@0.9'),
]

r2_matrix = np.zeros((len(metrics_for_corr), len(datasets)))

# Filter to the evaluation recall range [0.70, 0.95] used in the paper.
# This avoids skewing from extreme configurations outside the evaluation range.
RECALL_MIN, RECALL_MAX = 0.70, 0.95

for j, ds in enumerate(datasets):
    ds_rows = [r for r in rows if r['dataset'] == ds
               and RECALL_MIN <= float(r['recall@10']) <= RECALL_MAX]
    recalls = np.array([float(r['recall@10']) for r in ds_rows])
    for i, (key, label) in enumerate(metrics_for_corr):
        vals = np.array([float(r[key]) for r in ds_rows])
        if np.std(vals) == 0:
            r2_matrix[i, j] = 1.0
        else:
            r, p = pearsonr(recalls, vals)
            r2_matrix[i, j] = r**2

fig, ax = plt.subplots(figsize=(6, 3.5))
im = ax.imshow(r2_matrix, cmap='YlGnBu', aspect='auto', vmin=0.0, vmax=1.0)

ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(ds_labels, fontsize=9)
ax.set_yticks(range(len(metrics_for_corr)))
ax.set_yticklabels([label for _, label in metrics_for_corr], fontsize=9)

# Add text annotations
for i in range(len(metrics_for_corr)):
    for j in range(len(datasets)):
        val = r2_matrix[i, j]
        color = 'white' if val > 0.75 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('$r^2$ with Recall@10', fontsize=9)
ax.set_title('Correlation with average recall ($r^2$)', fontsize=10, pad=8)

# Add horizontal line separating IR metrics from robustness metrics
ax.axhline(y=4.5, color='black', linewidth=1.5, linestyle='-')

plt.tight_layout()
plt.savefig('figures/QueryFeatures/metric_correlation_r2.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/QueryFeatures/metric_correlation_r2.pdf")

# ============================================================
# Figure 2: Relative deviation from cross-family mean at fixed recall
# Shows what % each family deviates from the mean for each metric.
# Metrics that can't distinguish families show ~0% deviation.
# ============================================================
graph_algos = {'faiss_hnsw', 'diskann', 'zilliz'}
partition_algos = {'scann', 'puck'}

# All metrics, using failure rate (1-robustness) for low delta
metrics_to_show = [
    ('map@10', 'MAP@10', 'direct'),
    ('ndcg@10', 'NDCG@10', 'direct'),
    ('mrr@10', 'MRR@10', 'direct'),
    ('robustness@0.1', 'Fail rate\n($\\delta$=0.1)', 'failure'),
    ('robustness@0.3', 'Fail rate\n($\\delta$=0.3)', 'failure'),
    ('robustness@0.9', 'Rob.@0.9', 'direct'),
]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))

for ax_idx, (ds, ds_label) in enumerate(zip(datasets, ds_labels)):
    ax = axes[ax_idx]
    ds_rows = [r for r in rows if r['dataset'] == ds]

    # Bin by recall: 0.88-0.92
    target, eps = 0.90, 0.02
    near = [r for r in ds_rows if abs(float(r['recall@10']) - target) <= eps]

    if len(near) < 3:
        ax.set_title(f'{ds_label} (insufficient data)', fontsize=9)
        continue

    graph_rows = [r for r in near if r['algorithm'] in graph_algos]
    part_rows = [r for r in near if r['algorithm'] in partition_algos]

    # Compute relative deviation: (family_mean - overall_mean) / overall_mean * 100
    x = np.arange(len(metrics_to_show))
    width = 0.35
    graph_devs = []
    part_devs = []

    for key, label, mode in metrics_to_show:
        if mode == 'failure':
            g_vals = [1 - float(r[key]) for r in graph_rows]
            p_vals = [1 - float(r[key]) for r in part_rows]
        else:
            g_vals = [float(r[key]) for r in graph_rows]
            p_vals = [float(r[key]) for r in part_rows]

        g_mean = np.mean(g_vals) if g_vals else 0
        p_mean = np.mean(p_vals) if p_vals else 0
        overall = np.mean(g_vals + p_vals)

        if overall > 0:
            graph_devs.append((g_mean - overall) / overall * 100)
            part_devs.append((p_mean - overall) / overall * 100)
        else:
            graph_devs.append(0)
            part_devs.append(0)

    bars1 = ax.bar(x - width/2, graph_devs, width, label='Graph-based', color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x + width/2, part_devs, width, label='Partition-based', color='#3498db', alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=2.5, color='gray', linewidth=0.8, linestyle='--')

    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l, _ in metrics_to_show], fontsize=7)
    ax.set_title(f'{ds_label} (Recall@10 $\\approx$ 0.9)', fontsize=9)

    if ax_idx == 0:
        ax.set_ylabel('Deviation from mean (%)', fontsize=8)
        ax.legend(fontsize=7, loc='upper left')

    # Add value annotations on the larger bars
    for i, (gd, pd) in enumerate(zip(graph_devs, part_devs)):
        if abs(gd) > 5:
            ax.annotate(f'{gd:+.0f}%', xy=(i - width/2, gd),
                       fontsize=6, ha='center', va='bottom' if gd > 0 else 'top',
                       fontweight='bold')
        if abs(pd) > 5:
            ax.annotate(f'{pd:+.0f}%', xy=(i + width/2, pd),
                       fontsize=6, ha='center', va='bottom' if pd > 0 else 'top',
                       fontweight='bold')

plt.tight_layout()
plt.savefig('figures/QueryFeatures/metric_family_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/QueryFeatures/metric_family_comparison.pdf")

print("\nDone!")
