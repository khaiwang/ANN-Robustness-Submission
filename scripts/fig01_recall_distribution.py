#!/usr/bin/env python3
"""Fig 1 (Section 1): Recall distribution histogram on MSMARCO for ScaNN and DiskANN."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from common import load_cdf, COLORS, OUTPUT_DIR

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    cdf_data = load_cdf('msmarco-10M', k=10)

    algos = ['ScaNN', 'DiskANN']
    x_labels = [f'{i/10:.1f}' for i in range(11)]
    x = np.arange(len(x_labels))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(8, 3))

    for i, algo in enumerate(algos):
        cdf = cdf_data[algo]['cdf_values']
        # PDF: probability mass at each recall bin
        pdf = [cdf[j] - cdf[j + 1] for j in range(len(cdf) - 1)]
        pdf.append(cdf[-1])  # recall=1.0 bin

        bars = ax.bar(x + (i - 0.5) * bar_width, pdf, bar_width,
                      label=algo, color=COLORS[algo], edgecolor='white', linewidth=0.5)

        # Annotate recall=0.9 and recall=1.0 bars
        for idx in [9, 10]:
            ax.annotate(f'{pdf[idx]:.1%}', xy=(x[idx] + (i - 0.5) * bar_width, pdf[idx]),
                        ha='center', va='bottom', fontsize=7)

    # Highlight recall=0 bars
    for i, algo in enumerate(algos):
        cdf = cdf_data[algo]['cdf_values']
        pdf_0 = cdf[0] - cdf[1]
        rect = mpatches.FancyBboxPatch(
            (x[0] + (i - 0.5) * bar_width - bar_width / 2 - 0.02, -0.001),
            bar_width + 0.04, pdf_0 + 0.003,
            boxstyle="round,pad=0.01", edgecolor='red', facecolor='none', linewidth=1.5)
        ax.add_patch(rect)

    ax.set_xlabel('Recall@10', fontsize=11)
    ax.set_ylabel('Percentage of Queries', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylim(0, 0.10)
    ax.legend(fontsize=10)
    ax.tick_params(axis='y', labelsize=9)

    out = OUTPUT_DIR / 'fig01_recall_dist_msmarco.pdf'
    plt.tight_layout()
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
