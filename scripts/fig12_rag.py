#!/usr/bin/env python3
"""Fig 11 (Section 5.3): End-to-end RAG accuracy vs recall and robustness."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_rag, OUTPUT_DIR

COLORS = {
    'HNSW':    '#66CCEE',
    'IVFFlat': '#EE6677',
    'DiskANN': '#4477AA',
    'ScaNN':   '#FF9D23',
    'K-NN':    'grey',
}
MARKERS = {
    'HNSW': 'o', 'IVFFlat': '^', 'DiskANN': 's', 'ScaNN': 'v', 'K-NN': '*',
}

SCENARIOS = [
    ('naive', '(a) Naive RAG', 'Recall@10', 'Robustness-0.2@10'),
    ('agentic_searchr1', '(b) Agentic RAG (Search-R1)', 'Recall@5', 'Robustness-0.4@5'),
    ('agentic_qwen3', '(c) Agentic RAG (Qwen3-30B)', 'Recall@5', 'Robustness-0.4@5'),
]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    rag_data = load_rag()

    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    fig, axs = plt.subplots(3, 2, figsize=(10, 14))

    for row, (scenario, title, xlabel_left, xlabel_right) in enumerate(SCENARIOS):
        ax_left, ax_right = axs[row]
        algo_data = rag_data.get(scenario, {})

        for algo, points in algo_data.items():
            color = COLORS.get(algo, 'black')
            marker = MARKERS.get(algo, 'o')
            recalls = [p[0] for p in points]
            robs = [p[1] for p in points]
            accs = [p[2] for p in points]
            s = 100 if algo == 'K-NN' else 80
            ax_left.scatter(recalls, accs, marker=marker, color=color, label=algo, s=s, zorder=3)
            ax_right.scatter(robs, accs, marker=marker, color=color, label=algo, s=s, zorder=3)

        ax_left.set_xlabel(xlabel_left, fontsize=12)
        ax_right.set_xlabel(xlabel_right, fontsize=12)
        ax_left.set_ylabel('Accuracy', fontsize=12)
        ax_left.grid(True, alpha=0.3)
        ax_right.grid(True, alpha=0.3)
        ax_right.legend(fontsize=9, loc='lower right')

        # Row title
        ax_left.text(1.05, -0.35, title, transform=ax_left.transAxes,
                     fontsize=12, ha='center')

    out = OUTPUT_DIR / 'fig12_rag.pdf'
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.45)
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
