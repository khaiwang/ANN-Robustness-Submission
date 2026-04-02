#!/usr/bin/env python3
"""Fig 8 (Section 5.2): Three-way tradeoff (Zilliz + ScaNN) on text2image-10M."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_aggregate, COLORS, MARKERS, OUTPUT_DIR

TRADEOFF_ALGOS = ['Zilliz', 'ScaNN']
RECALL_THRESHOLDS = [0.85, 0.9]
ROB_THRESHOLDS = [0.95, 0.99]
QPS_THRESHOLDS = [15000, 30000]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = load_aggregate('text2image-10M', k=10)
    results = [r for r in results if r['algo'] in TRADEOFF_ALGOS]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plt.rcParams.update({'font.size': 11})

    # Panel (a): Fix recall, show Robustness-0.3 vs QPS
    ax = axes[0]
    for thr_idx, recall_thr in enumerate(RECALL_THRESHOLDS):
        ls = '--' if thr_idx == 0 else '-'
        for algo in TRADEOFF_ALGOS:
            filtered = [r for r in results if r['algo'] == algo and r['recall'] >= recall_thr]
            robs = [r['robustness'][0.3] for r in filtered]
            qps = [r['qps'] for r in filtered]
            label = f'{algo} (recall>={recall_thr})' if thr_idx == 0 else None
            ax.scatter(robs, qps, color=COLORS[algo], marker=MARKERS[algo],
                       s=30, alpha=0.7, label=algo if thr_idx == 0 else None)
    ax.set_xlabel('Robustness-0.3@10', fontsize=10)
    ax.set_ylabel('QPS', fontsize=10)
    ax.set_title('(a) Fix: Average Recall', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (b): Fix robustness, show Recall vs QPS
    ax = axes[1]
    for thr_idx, rob_thr in enumerate(ROB_THRESHOLDS):
        for algo in TRADEOFF_ALGOS:
            filtered = [r for r in results if r['algo'] == algo and r['robustness'][0.3] >= rob_thr]
            recalls = [r['recall'] for r in filtered]
            qps = [r['qps'] for r in filtered]
            ax.scatter(recalls, qps, color=COLORS[algo], marker=MARKERS[algo],
                       s=30, alpha=0.7, label=algo if thr_idx == 0 else None)
    ax.set_xlabel('Average Recall@10', fontsize=10)
    ax.set_ylabel('QPS', fontsize=10)
    ax.set_title('(b) Fix: Robustness-0.3', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (c): Fix QPS, show Recall vs Robustness
    ax = axes[2]
    for thr_idx, qps_thr in enumerate(QPS_THRESHOLDS):
        for algo in TRADEOFF_ALGOS:
            filtered = [r for r in results if r['algo'] == algo and r['qps'] >= qps_thr]
            recalls = [r['recall'] for r in filtered]
            robs = [r['robustness'][0.3] for r in filtered]
            ax.scatter(recalls, robs, color=COLORS[algo], marker=MARKERS[algo],
                       s=30, alpha=0.7, label=algo if thr_idx == 0 else None)
    ax.set_xlabel('Average Recall@10', fontsize=10)
    ax.set_ylabel('Robustness-0.3@10', fontsize=10)
    ax.set_title('(c) Fix: QPS', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out = OUTPUT_DIR / 'fig09_tradeoff_zilliz_scann.pdf'
    plt.tight_layout()
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
