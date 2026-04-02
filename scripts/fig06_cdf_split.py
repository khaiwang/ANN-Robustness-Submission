#!/usr/bin/env python3
"""Fig 5 (Section 5.1): CDF split plot for text2image-10M, K=10 and K=100."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import load_cdf, COLORS, MARKERS, ALGO_ORDER, CDF_DELTAS, OUTPUT_DIR


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    cdf_k10 = load_cdf('text2image-10M', k=10)
    cdf_k100 = load_cdf('text2image-10M', k=100)
    deltas = np.array(CDF_DELTAS)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    plt.rcParams.update({'font.size': 12})

    for row, (cdf_data, k_label) in enumerate([(cdf_k10, 'K=10'), (cdf_k100, 'K=100')]):
        ax_left = axes[row][0]
        ax_right = axes[row][1]

        for algo in ALGO_ORDER:
            if algo not in cdf_data:
                continue
            vals = np.array(cdf_data[algo]['cdf_values'])
            color = COLORS[algo]
            marker = MARKERS[algo]

            # Left panel: delta <= 0.7
            mask_left = deltas <= 0.7 + 1e-9
            ax_left.plot(deltas[mask_left], vals[mask_left],
                         color=color, marker=marker, markersize=5, label=algo, linewidth=1.5)

            # Right panel: delta >= 0.7
            mask_right = deltas >= 0.7 - 1e-9
            ax_right.plot(deltas[mask_right], vals[mask_right],
                          color=color, marker=marker, markersize=5, label=algo, linewidth=1.5)

        ax_left.set_xlim(-0.02, 0.72)
        ax_left.set_ylim(0.90, 1.005)
        ax_right.set_xlim(0.68, 1.02)
        if row == 0:
            ax_right.set_ylim(0.4, 1.02)
        else:
            ax_right.set_ylim(0.1, 1.02)

        ax_left.set_ylabel(f'Robustness ({k_label})', fontsize=11)
        ax_left.grid(True, alpha=0.3)
        ax_right.grid(True, alpha=0.3)

        if row == 1:
            ax_left.set_xlabel(r'$\delta$', fontsize=11)
            ax_right.set_xlabel(r'$\delta$', fontsize=11)

        ax_right.legend(fontsize=8, loc='lower left')

    out = OUTPUT_DIR / 'fig06_robustness_cdf_split.pdf'
    plt.tight_layout()
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
