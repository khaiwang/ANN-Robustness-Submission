#!/usr/bin/env python3
"""Generate RAG figure: 3 rows × 2 cols (Naive, Agentic Search-R1, Agentic Qwen3-30B)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Colors matching our palette
c_hnsw = '#66CCEE'
c_ivf = '#EE6677'
c_diskann = '#4477AA'
c_scann = '#FF9D23'
c_knn = 'grey'

# ============================================================
# 1. NAIVE RAG (from Data.ipynb)
# ============================================================
hnsw_naive = [
    (0.861, 0.926, 0.701),  # recall, rob@0.2, accuracy
    (0.881, 0.941, 0.713),
    (0.900, 0.957, 0.726),
    (0.930, 0.982, 0.749),
]
ivf_naive = [
    (0.851, 0.973, 0.740),
    (0.882, 0.980, 0.749),
    (0.901, 0.987, 0.751),
    (0.928, 0.994, 0.757),
]
diskann_naive = [
    (0.885, 0.933, 0.711),
    (0.873, 0.922, 0.696),
    (0.901, 0.943, 0.714),
    (0.919, 0.956, 0.726),
    (0.931, 0.964, 0.733),
    (0.946, 0.974, 0.744),
    (0.960, 0.983, 0.749),
]
scann_naive = [
    (0.876, 0.973, 0.740),
    (0.898, 0.983, 0.749),
    (0.933, 0.993, 0.755),
    (0.944, 0.995, 0.757),
    (0.950, 0.997, 0.758),
    (0.954, 0.997, 0.761),
]
knn_naive = (1.0, 1.0, 0.767)

# ============================================================
# 2. AGENTIC RAG - Search-R1 Qwen2.5-7B (from Data.ipynb)
# ============================================================
hnsw_agentic_sr1 = [
    (0.801, 0.871, 0.483),  # recall@5, rob@0.4, accuracy
    (0.847, 0.923, 0.495),
    (0.901, 0.954, 0.505),
]
ivf_agentic_sr1 = [
    (0.802, 0.922, 0.497),
    (0.853, 0.959, 0.506),
    (0.898, 0.976, 0.510),
]
knn_agentic_sr1 = (1.0, 1.0, 0.526)

# ============================================================
# 3. AGENTIC RAG - Qwen3-30B-A3B (DocOnly accuracy — excludes internal knowledge)
# Recall/robustness from tuning query vectors; accuracy = DocOnly from doc-validity analysis
# ============================================================
hnsw_agentic_q3 = [
    (0.83, 0.906, 0.506),  # hnsw_ef30: DocOnly=50.6%
    (0.86, 0.928, 0.514),  # hnsw_ef40: DocOnly=51.4%
    (0.91, 0.964, 0.530),  # hnsw_ef80: DocOnly=53.0%
]
ivf_agentic_q3 = [
    (0.81, 0.926, 0.506),  # ivf_np60:  DocOnly=50.6%
    (0.85, 0.952, 0.520),  # ivf_np100: DocOnly=52.0%
    (0.91, 0.977, 0.542),  # ivf_np200: DocOnly=54.2%
]
knn_agentic_q3 = (1.0, 1.0, 0.555)  # knn_flat: DocOnly=55.5%

# ============================================================
# Plot
# ============================================================
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
fig, axs = plt.subplots(3, 2, figsize=(10, 14))

def plot_row(ax_left, ax_right, datasets, knn_pt, x_label_left, x_label_right, y_label):
    for data, color, marker, label in datasets:
        recalls = [d[0] for d in data]
        robs = [d[1] for d in data]
        accs = [d[2] for d in data]
        ax_left.scatter(recalls, accs, marker=marker, color=color, label=label, s=80, zorder=3)
        ax_right.scatter(robs, accs, marker=marker, color=color, label=label, s=80, zorder=3)
    ax_left.scatter(knn_pt[0], knn_pt[2], marker='*', color=c_knn, label='K-NN', s=100, zorder=3)
    ax_right.scatter(knn_pt[1], knn_pt[2], marker='*', color=c_knn, label='K-NN', s=100, zorder=3)
    ax_left.set_xlabel(x_label_left, fontsize=12)
    ax_right.set_xlabel(x_label_right, fontsize=12)
    ax_left.set_ylabel(y_label, fontsize=12)
    ax_left.grid(True, alpha=0.3)
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=9, loc='lower right')

# Row (a): Naive RAG
plot_row(axs[0][0], axs[0][1],
    [(hnsw_naive, c_hnsw, 'o', 'HNSW'),
     (ivf_naive, c_ivf, '^', 'IVFFlat'),
     (diskann_naive, c_diskann, 's', 'DiskANN'),
     (scann_naive, c_scann, 'v', 'ScaNN')],
    knn_naive, 'Recall@10', 'Robustness-0.2@10', 'Accuracy')

# Row (b): Agentic RAG - Search-R1
plot_row(axs[1][0], axs[1][1],
    [(hnsw_agentic_sr1, c_hnsw, 'o', 'HNSW'),
     (ivf_agentic_sr1, c_ivf, '^', 'IVFFlat')],
    knn_agentic_sr1, 'Recall@5', 'Robustness-0.4@5', 'Accuracy')

# Row (c): Agentic RAG - Qwen3-30B
plot_row(axs[2][0], axs[2][1],
    [(hnsw_agentic_q3, c_hnsw, 'o', 'HNSW'),
     (ivf_agentic_q3, c_ivf, '^', 'IVFFlat')],
    knn_agentic_q3, 'Recall@5', 'Robustness-0.4@5', 'Accuracy')

# Row titles — bottom center of each row, no bold
for row, title in enumerate(['(a) Naive RAG', '(b) Agentic RAG (Search-R1)', '(c) Agentic RAG (Qwen3-30B)']):
    axs[row][0].text(1.05, -0.35, title,
                     transform=axs[row][0].transAxes,
                     fontsize=12, ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.45)
plt.savefig('../Robustness-VLDB-Revision/figures/rag.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()
print("Saved: ../Robustness-VLDB-Revision/figures/rag.pdf")
