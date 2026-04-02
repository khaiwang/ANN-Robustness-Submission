#!/usr/bin/env python3
"""Fig 7 (Section 5.1): 5-panel recall vs robustness scatter on text2image-10M."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import load_aggregate, COLORS, MARKERS, ALGO_ORDER, ROBUSTNESS_DELTAS, OUTPUT_DIR


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = load_aggregate('text2image-10M', k=10)

    fig, axes = plt.subplots(1, len(ROBUSTNESS_DELTAS), figsize=(18, 3.2), sharey=False)
    plt.rcParams.update({'font.size': 11})

    for panel_idx, delta in enumerate(ROBUSTNESS_DELTAS):
        ax = axes[panel_idx]

        for algo in ALGO_ORDER:
            algo_results = [r for r in results if r['algo'] == algo and r['recall'] >= 0.7]
            if not algo_results:
                continue

            # Deduplicate by recall bin (keep max robustness per bin)
            by_bin = {}
            for r in algo_results:
                rbin = round(r['recall'], 2)
                rob = r['robustness'][delta]
                if rbin not in by_bin or rob > by_bin[rbin]:
                    by_bin[rbin] = rob

            recalls = sorted(by_bin.keys())
            robs = [by_bin[rc] for rc in recalls]

            ax.plot(recalls, robs, '-', color=COLORS[algo], marker=MARKERS[algo],
                    markersize=4, label=algo, linewidth=1.2)

        ax.set_xlabel('Average Recall@10', fontsize=10)
        ax.set_title(f'Robustness-{delta}@10', fontsize=10)
        ax.grid(True, alpha=0.3)
        if panel_idx == 0:
            ax.set_ylabel('Robustness', fontsize=10)
        if panel_idx == len(ROBUSTNESS_DELTAS) - 1:
            ax.legend(fontsize=7, loc='lower right')

    out = OUTPUT_DIR / 'fig08_recall_robustness_5panel.pdf'
    plt.tight_layout()
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
