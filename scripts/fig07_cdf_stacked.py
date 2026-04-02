#!/usr/bin/env python3
"""Fig 6 (Section 5.1): Stacked CDF for MSSPACEV, DEEP, and MSMARCO (K=10)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import load_cdf, COLORS, MARKERS, ALGO_ORDER, CDF_DELTAS, OUTPUT_DIR

STACKED_DATASETS = [
    ('msspacev-10M', 'MSSPACEV-10M'),
    ('deep-10M', 'DEEP-10M'),
    ('msmarco-10M', 'MSMARCO'),
]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    deltas = np.array(CDF_DELTAS)

    fig, axes = plt.subplots(len(STACKED_DATASETS), 2, figsize=(7, 10))
    plt.rcParams.update({'font.size': 11})

    for row, (ds_name, ds_label) in enumerate(STACKED_DATASETS):
        cdf_data = load_cdf(ds_name, k=10)
        ax_left = axes[row][0]
        ax_right = axes[row][1]

        for algo in ALGO_ORDER:
            if algo not in cdf_data:
                continue
            vals = np.array(cdf_data[algo]['cdf_values'])
            color = COLORS[algo]
            marker = MARKERS[algo]

            mask_left = deltas <= 0.7 + 1e-9
            ax_left.plot(deltas[mask_left], vals[mask_left],
                         color=color, marker=marker, markersize=4, label=algo, linewidth=1.3)

            mask_right = deltas >= 0.7 - 1e-9
            ax_right.plot(deltas[mask_right], vals[mask_right],
                          color=color, marker=marker, markersize=4, label=algo, linewidth=1.3)

        ax_left.set_xlim(-0.02, 0.72)
        ax_left.set_ylim(0.90, 1.005)
        ax_right.set_xlim(0.68, 1.02)
        ax_right.set_ylim(0.1, 1.02)
        ax_left.set_ylabel(ds_label, fontsize=10)
        ax_left.grid(True, alpha=0.3)
        ax_right.grid(True, alpha=0.3)

        if row == len(STACKED_DATASETS) - 1:
            ax_left.set_xlabel(r'$\delta$', fontsize=10)
            ax_right.set_xlabel(r'$\delta$', fontsize=10)

        if row == 0:
            ax_right.legend(fontsize=7, loc='lower left')

    out = OUTPUT_DIR / 'fig07_robustness_cdf_stacked.pdf'
    plt.tight_layout()
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
