#!/usr/bin/env python3
"""
Replot Figures 1, 6, 7+8 (merged), 9 (skip), 10 for VLDB revision.
Figures go to Robustness-VLDB-Revision/figures/
"""

import sys, os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark.datasets import DATASETS
from benchmark.plotting.metrics import get_recall_values

# ============================================================
# Color Palette — Tableau-inspired, colorblind-friendly
# ============================================================
# Paul Tol "bright" — Graph: cool, Partition: warm
COLORS = {
    'HNSW':    '#66CCEE',  # cyan (swapped with DiskANN)
    'DiskANN': '#4477AA',  # blue (swapped with HNSW)
    'Zilliz':  '#AA3377',  # magenta
    'IVFFlat': '#EE6677',  # coral
    'ScaNN':   '#FF9D23',  # amber/orange (original fig 1 color)
    'Puck':    '#44AA77',  # medium teal-green
}

MARKERS = {
    'HNSW':    'o',
    'DiskANN': 's',
    'Zilliz':  'D',
    'IVFFlat': '^',
    'ScaNN':   'v',
    'Puck':    'P',
}

ALGO_MAP = {
    'faiss_hnsw': 'HNSW',
    'diskann': 'DiskANN',
    'zilliz': 'Zilliz',
    'faiss-ivf': 'IVFFlat',
    'scann': 'ScaNN',
    'puck': 'Puck',
}

ALGO_ORDER = ['HNSW', 'DiskANN', 'Zilliz', 'IVFFlat', 'ScaNN', 'Puck']

RESULTS_DIR = 'results/neurips23/ood'
PAPER_DIR = '../Robustness-VLDB-Revision/figures'
REPLOT_DIR = '../Robustness-VLDB-Revision/figures'


def load_all_results(dataset_name, count=10, results_dir=None):
    ds = DATASETS[dataset_name]()
    gt = ds.get_groundtruth()
    results = []
    if results_dir is None:
        results_dir = RESULTS_DIR

    ds_dir = os.path.join(results_dir, dataset_name, str(count))
    if not os.path.exists(ds_dir):
        return results

    for algo_dir in sorted(os.listdir(ds_dir)):
        display_name = ALGO_MAP.get(algo_dir)
        if display_name is None:
            continue

        algo_path = os.path.join(ds_dir, algo_dir)
        for fname in sorted(os.listdir(algo_path)):
            if not fname.endswith('.hdf5'):
                continue
            fpath = os.path.join(algo_path, fname)
            try:
                with h5py.File(fpath, 'r') as f:
                    neighbors = np.array(f['neighbors'])
                    search_times = np.array(f.attrs.get('search_times', [f.attrs.get('best_search_time', 1.0)]))
                    best_time = np.min(search_times)
                    n_queries = neighbors.shape[0]
                    qps = n_queries / best_time if best_time > 0 else 0

                    recall_mean, _, recalls, _ = get_recall_values(gt, neighbors, count)
                    recalls_norm = recalls / float(count)

                    deltas = [0.1, 0.3, 0.5, 0.7, 0.9]
                    robustness = {d: float(np.mean(recalls_norm >= d)) for d in deltas}

                    # Use exact fractions to avoid floating-point boundary errors
                    # (e.g. 3/10 >= 0.30000000000000004 is False)
                    cdf_deltas = np.array([i/10.0 for i in range(11)])
                    cdf_values = [float(np.mean(recalls >= d * count - 0.5)) for d in cdf_deltas]
                    cdf_values[0] = 1.0  # δ=0: all queries have recall >= 0

                    results.append({
                        'algo': display_name,
                        'config': fname.replace('.hdf5', ''),
                        'recall': float(recall_mean),
                        'robustness': robustness,
                        'qps': qps,
                        'cdf_deltas': cdf_deltas.tolist(),
                        'cdf_values': cdf_values,
                        'per_query_recalls': recalls_norm,
                    })
            except Exception as e:
                print(f"  Error loading {fpath}: {e}")

    print(f"Loaded {len(results)} configurations for {dataset_name}")
    return results


def find_config_at_recall(results, target_recall=0.9, tolerance=0.02):
    """For each algo, find config with recall closest to target within tolerance."""
    by_algo = defaultdict(list)
    for r in results:
        by_algo[r['algo']].append(r)

    selected = {}
    for algo, configs in by_algo.items():
        # Prefer configs within tolerance; pick closest to target
        candidates = sorted(configs, key=lambda c: abs(c['recall'] - target_recall))
        best = candidates[0]
        if abs(best['recall'] - target_recall) > tolerance:
            print(f"  WARNING: {algo} best recall={best['recall']:.4f}, target={target_recall}")
        else:
            print(f"  {algo}: recall={best['recall']:.4f}")
        selected[algo] = best
    return selected


# ============================================================
# Figure 1: Recall distribution on MSMARCO
# ============================================================

def plot_figure1(all_results, output_path):
    font = 22
    plt.rcParams.update({'font.size': font, 'font.family': 'serif'})

    # Strictly pick configs at recall ≈ 0.9
    selected = find_config_at_recall(all_results['msmarco-10M'], 0.9, tolerance=0.02)

    algos_to_plot = ['ScaNN', 'DiskANN']
    colors_plot = [COLORS['ScaNN'], COLORS['DiskANN']]
    hatches = ['/', None]

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    x_labels = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    bar_width = 0.4
    x = np.arange(len(x_labels))

    pdfs = {}
    for i, algo in enumerate(algos_to_plot):
        if algo not in selected:
            continue
        config = selected[algo]
        print(f"  Fig1 {algo}: recall={config['recall']:.4f}")
        cdf = config['cdf_values']
        pdf = [cdf[j] - cdf[j+1] for j in range(len(cdf)-1)]
        pdf.append(cdf[-1])  # mass at recall=1.0
        pdfs[algo] = pdf

        offset = -bar_width/2 if i == 0 else bar_width/2
        ax.bar(x + offset, pdf, bar_width, label=algo,
               color=colors_plot[i], edgecolor='black', zorder=2,
               hatch=hatches[i])

    # Annotate bars exceeding ylim at top
    for i, algo in enumerate(algos_to_plot):
        if algo not in pdfs:
            continue
        # recall=0.9 — only label if exceeds ylim
        val09 = pdfs[algo][9]
        if val09 > 0.10:
            # Position to the left of the 0.9 bars
            lx = x[9] - bar_width * 1.5
            ax.annotate(f'{val09:.2f}', xy=(lx, 0.1),
                        xytext=(0, 0.4), textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="grey", alpha=0.6),
                        fontsize=font)
        # recall=1.0 — spread wide
        val10 = pdfs[algo][10]
        if val10 > 0.10:
            lx = x[10] - bar_width * 0.8 if i == 0 else x[10] + bar_width * 1.8
            ax.annotate(f'{val10:.2f}', xy=(lx, 0.1),
                        xytext=(0, 0.4), textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="grey", alpha=0.6),
                        fontsize=font)

    # Red highlight on recall=0
    highlight_x = x[0]
    highlight_width = bar_width * 2.2
    max_zero = max(pdfs[a][0] for a in algos_to_plot if a in pdfs)
    highlight_height = max_zero + 0.005

    rect = plt.Rectangle(
        (highlight_x - highlight_width/2 - 0.01, 0),
        highlight_width + 0.02, highlight_height,
        edgecolor="red", linewidth=3, facecolor="none")
    ax.add_patch(rect)

    # Annotate recall=0 values — match old notebook style
    for i, algo in enumerate(algos_to_plot):
        if algo not in pdfs:
            continue
        val = pdfs[algo][0]
        label_x = x[0] + (bar_width/2)*2.3 if i == 1 else x[0] - (bar_width/2)*2.3
        ax.annotate(f'{val:.3f}',
                    xy=(label_x, highlight_height),
                    xytext=(0, 0.4), textcoords='offset points',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="grey", alpha=0.6),
                    fontsize=font)

    ax.set_ylim(0, 0.10)
    ax.set_yticks(np.arange(0, 0.11, 0.02))
    ax.set_xticks(x, x_labels, fontsize=font)
    ax.set_xlabel('Recall', fontsize=font)
    ax.set_ylabel('Distribution of Recall', fontsize=font)
    plt.grid(zorder=1, axis='y')
    plt.legend(fontsize=font)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Figure 6: Text2Image CDF split (2-panel: low-δ zoomed, high-δ)
# Matches original format: left panel δ=0-0.7 y=0.90-1.00, right panel δ=0.7-1.0 y=0.4-1.0
# ============================================================

def plot_split_cdf(results_k10, output_path, results_k100=None):
    """2x2 split CDF: top row K=10, bottom row K=100.
    Left: δ=0-0.7 zoomed (y=0.90-1.00), Right: δ=0.7-1.0 (y=0.4-1.0)."""
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    selected_k10 = find_config_at_recall(results_k10, 0.9, tolerance=0.02)
    selected_k100 = find_config_at_recall(results_k100, 0.9, tolerance=0.03) if results_k100 else selected_k10
    deltas = np.array([i/10.0 for i in range(11)])

    row_labels = [('(a) K=10, δ=0-0.7', '(b) K=10, δ=0.7-1.0'),
                  ('(c) K=100, δ=0-0.7', '(d) K=100, δ=0.7-1.0')]
    row_selected = [selected_k10, selected_k100]

    for row in range(2):
        selected = row_selected[row]
        for algo_name in ALGO_ORDER:
            if algo_name not in selected:
                continue
            config = selected[algo_name]
            cdf = np.array(config['cdf_values'])

            # Left panel: δ = 0 to 0.7
            low_mask = deltas <= 0.7 + 1e-9
            axes[row, 0].plot(deltas[low_mask], cdf[low_mask],
                        label=algo_name,
                        color=COLORS[algo_name],
                        marker=MARKERS[algo_name],
                        markersize=7, linewidth=1.8)

            # Right panel: δ = 0.7 to 1.0
            high_mask = deltas >= 0.7 - 1e-9
            axes[row, 1].plot(deltas[high_mask], cdf[high_mask],
                        label=algo_name,
                        color=COLORS[algo_name],
                        marker=MARKERS[algo_name],
                        markersize=7, linewidth=1.8)

        # Left panel formatting
        axes[row, 0].set_xlabel(r'$\delta$', fontsize=12)
        axes[row, 0].set_ylabel(r'Robustness-$\delta$@K', fontsize=12)
        axes[row, 0].set_xlim(-0.02, 0.72)
        axes[row, 0].set_ylim(0.90, 1.005)
        axes[row, 0].set_xticks(np.arange(0, 0.8, 0.1))
        axes[row, 0].yaxis.set_major_locator(MultipleLocator(0.02))
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].set_title(row_labels[row][0], y=-0.28, fontsize=10)

        # Right panel formatting
        axes[row, 1].set_xlabel(r'$\delta$', fontsize=12)
        axes[row, 1].set_xlim(0.68, 1.02)
        axes[row, 1].set_ylim(0.1 if row == 1 else 0.4, 1.02)  # lower ylim for K=100
        axes[row, 1].set_xticks([0.7, 0.8, 0.9, 1.0])
        axes[row, 1].yaxis.set_major_locator(MultipleLocator(0.1))
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].set_title(row_labels[row][1], y=-0.28, fontsize=10)

    # Legend above figure in one row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=9,
              bbox_to_anchor=(0.5, 1.03), frameon=True,
              columnspacing=0.8, handletextpad=0.3, handlelength=1.5)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Figure 7+8 merged: Stacked CDF for MSSPACEV, DEEP, MSMARCO
# ============================================================

def plot_stacked_cdf_3(all_results, output_path):
    """3 datasets × 2 split panels, single column.
    Left: δ=0-0.7 zoomed (y=0.90-1.00), Right: δ=0.7-1.0 (y=0.4-1.0)."""
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})

    datasets = ['msspacev-10M', 'deep-10M', 'msmarco-10M']
    row_titles = ['MSSPACEV', 'DEEP', 'MSMARCO']

    fig, axes = plt.subplots(3, 2, figsize=(3.5, 3.8))
    plt.subplots_adjust(hspace=0.55, wspace=0.35)

    deltas = np.array([i/10.0 for i in range(11)])

    for row, (ds_name, title) in enumerate(zip(datasets, row_titles)):
        results = all_results[ds_name]
        print(f"  CDF for {ds_name}:")
        selected = find_config_at_recall(results, 0.9, tolerance=0.02)

        for algo_name in ALGO_ORDER:
            if algo_name not in selected:
                continue
            config = selected[algo_name]
            cdf = np.array(config['cdf_values'])

            # Left: δ=0-0.7, zoomed y
            low_mask = deltas <= 0.7 + 1e-9
            axes[row, 0].plot(deltas[low_mask], cdf[low_mask],
                        label=algo_name,
                        color=COLORS[algo_name],
                        marker=MARKERS[algo_name],
                        markersize=2, linewidth=0.7)

            # Right: δ=0.7-1.0
            high_mask = deltas >= 0.7 - 1e-9
            axes[row, 1].plot(deltas[high_mask], cdf[high_mask],
                        label=algo_name,
                        color=COLORS[algo_name],
                        marker=MARKERS[algo_name],
                        markersize=2, linewidth=0.7)

        # Left panel
        axes[row, 0].set_xlim(-0.02, 0.72)
        axes[row, 0].set_ylim(0.90, 1.005)
        axes[row, 0].set_xticks(np.arange(0, 0.8, 0.2))
        axes[row, 0].yaxis.set_major_locator(MultipleLocator(0.02))
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].tick_params(labelsize=6.5)
        axes[row, 0].set_ylabel(r'Robustness-$\delta$@10', fontsize=7)
        for spine in axes[row, 0].spines.values():
            spine.set_linewidth(0.5)

        # Right panel
        axes[row, 1].set_xlim(0.68, 1.02)
        axes[row, 1].set_ylim(0.35, 1.02)
        axes[row, 1].set_xticks([0.7, 0.8, 0.9, 1.0])
        axes[row, 1].yaxis.set_major_locator(MultipleLocator(0.1))
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].tick_params(labelsize=6.5)
        for spine in axes[row, 1].spines.values():
            spine.set_linewidth(0.5)

        # Row title at bottom center spanning both panels
        axes[row, 0].text(1.18, -0.42, f'({chr(97+row)}) {title}',
                         transform=axes[row, 0].transAxes,
                         fontsize=7, ha='center')

    # x-labels only on bottom row
    axes[2, 0].set_xlabel(r'$\delta$', fontsize=8)
    axes[2, 1].set_xlabel(r'$\delta$', fontsize=8)

    # Legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=6.5,
              bbox_to_anchor=(0.5, 1.01), frameon=True,
              columnspacing=0.6, handletextpad=0.3, handlelength=1.5)

    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Figure 9: Tradeoff — only Zilliz & ScaNN, linear QPS
# ============================================================

def plot_tradeoff_zilliz_scann(results, output_path):
    """3-panel tradeoff: only Zilliz and ScaNN, matching paper logic."""
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))

    algos = ['Zilliz', 'ScaNN']
    algo_colors = {a: COLORS[a] for a in algos}
    algo_markers = {'Zilliz': 'o', 'ScaNN': '.'}

    linestyles_thresh = [('--', 1.2), ('-', 2.0)]

    # (a) Fix recall >= 0.85 / 0.9, plot Robustness@0.3 vs QPS
    recall_thresholds = [0.85, 0.9]
    ax = axes[0]
    for ti, (thresh, (ls, lw)) in enumerate(zip(recall_thresholds, linestyles_thresh)):
        for algo in algos:
            algo_results = [r for r in results if r['algo'] == algo and r['recall'] >= thresh]
            if not algo_results:
                continue
            algo_results.sort(key=lambda r: r['robustness'][0.3])
            rob = [r['robustness'][0.3] for r in algo_results]
            qps = [r['qps'] for r in algo_results]
            label = algo if ti == 0 else None
            ax.plot(rob, qps, color=algo_colors[algo], marker=algo_markers[algo],
                   markersize=5, linewidth=lw, linestyle=ls, label=label)
    ax.set_xlabel('Robustness', fontsize=11)
    ax.set_ylabel('QPS', fontsize=11)
    ax.set_title('(a) Limiting Average Recall > 0.85 and > 0.9', fontsize=10, y=-0.30)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # (b) Fix Robustness@0.3 >= 0.95 / 0.99, plot Recall vs QPS
    rob_thresholds = [0.95, 0.99]
    ax = axes[1]
    for ti, (thresh, (ls, lw)) in enumerate(zip(rob_thresholds, linestyles_thresh)):
        for algo in algos:
            algo_results = [r for r in results if r['algo'] == algo and r['robustness'][0.3] >= thresh]
            if not algo_results:
                continue
            algo_results.sort(key=lambda r: r['recall'])
            recall = [r['recall'] for r in algo_results]
            qps = [r['qps'] for r in algo_results]
            ax.plot(recall, qps, color=algo_colors[algo], marker=algo_markers[algo],
                   markersize=5, linewidth=lw, linestyle=ls)
    ax.set_xlabel('Average Recall', fontsize=11)
    ax.set_ylabel('QPS', fontsize=11)
    ax.set_title(r'(b) Limiting Robustness-0.3@10 > 0.95 and > 0.99', fontsize=10, y=-0.30)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    ax.grid(True, alpha=0.3)

    # (c) Fix QPS >= 15K / 30K (adjusted for new machine), plot Recall vs Robustness@0.3
    qps_thresholds = [15000, 30000]
    ax = axes[2]
    for ti, (thresh, (ls, lw)) in enumerate(zip(qps_thresholds, linestyles_thresh)):
        for algo in algos:
            algo_results = [r for r in results if r['algo'] == algo and r['qps'] >= thresh]
            if not algo_results:
                continue
            algo_results.sort(key=lambda r: r['recall'])
            recall = [r['recall'] for r in algo_results]
            rob = [r['robustness'][0.3] for r in algo_results]
            ax.plot(recall, rob, color=algo_colors[algo], marker=algo_markers[algo],
                   markersize=5, linewidth=lw, linestyle=ls)
    ax.set_xlabel('Average Recall', fontsize=11)
    ax.set_ylabel('Robustness', fontsize=11)
    ax.set_title('(c) Limiting QPS > 15K and > 30K', fontsize=10, y=-0.30)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Figure 10: text2image robustness-recall (5 panels, figure*)
# ============================================================

def plot_robustness_recall_5panel(results, output_path):
    """5-panel robustness-recall for δ=0.1, 0.3, 0.5, 0.7, 0.9."""
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.2))
    deltas = [0.1, 0.3, 0.5, 0.7, 0.9]
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

    for di, (delta, label) in enumerate(zip(deltas, panel_labels)):
        ax = axes[di]
        for algo_name in ALGO_ORDER:
            algo_results = [r for r in results if r['algo'] == algo_name and r['recall'] >= 0.7]
            if not algo_results:
                continue

            # Deduplicate by 0.01 recall bins
            bins = {}
            for r in algo_results:
                b = round(r['recall'], 2)
                if b not in bins or r['robustness'][delta] > bins[b]['robustness'][delta]:
                    bins[b] = r
            sorted_bins = sorted(bins.values(), key=lambda r: r['recall'])

            recalls = [r['recall'] for r in sorted_bins]
            rob_values = [r['robustness'][delta] for r in sorted_bins]
            ax.plot(recalls, rob_values,
                   label=algo_name,
                   color=COLORS[algo_name],
                   marker=MARKERS[algo_name],
                   markersize=6, linewidth=1.5)

        ax.set_xlabel('Average Recall@10', fontsize=10)
        ax.set_ylabel(f'Robustness-{delta}@10', fontsize=10)
        ax.set_title(f'{label} Robustness-{delta}@10', fontsize=10, y=-0.35)
        ax.set_xlim(0.69, 0.96)
        ax.set_xticks(np.arange(0.70, 0.96, 0.05))
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)

    # Legend on last panel
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Main
# ============================================================

FIGURE_REGISTRY = {
    'fig1':  'Figure 1: recall distribution (MSMARCO)',
    'fig6':  'Figure 6: text2image CDF split (K=10/K=100)',
    'fig7':  'Figure 7+8: stacked CDF (msspacev, deep, msmarco)',
    'fig9':  'Figure 9: tradeoff (Zilliz+ScaNN)',
    'fig10': 'Figure 10: robustness-recall 5-panel (text2image)',
    'rag':   'RAG figure: 3 rows (naive, agentic SR1, agentic Q3)',
}

# Datasets needed per figure
FIGURE_DATASETS = {
    'fig1':  ['msmarco-10M'],
    'fig6':  ['text2image-10M'],
    'fig7':  ['msspacev-10M', 'deep-10M', 'msmarco-10M'],
    'fig9':  ['text2image-10M'],
    'fig10': ['text2image-10M'],
    'rag':   [],  # uses hardcoded data
}


def run_figure(fig_name, all_results=None):
    """Generate a single figure. Loads only required datasets."""
    os.makedirs(os.path.join(REPLOT_DIR, 'OverallEval'), exist_ok=True)
    os.makedirs(os.path.join(REPLOT_DIR, 'Distribution-Motivation'), exist_ok=True)

    if all_results is None:
        all_results = {}
    # Load needed datasets
    for ds in FIGURE_DATASETS.get(fig_name, []):
        if ds not in all_results:
            print(f"Loading {ds}...")
            all_results[ds] = load_all_results(ds)

    if fig_name == 'fig1':
        print("Figure 1 (recall distribution MSMARCO):")
        plot_figure1(all_results,
            os.path.join(REPLOT_DIR, 'Distribution-Motivation', 'recall_dist_msmarco.pdf'))

    elif fig_name == 'fig6':
        print("Loading K=100 data...")
        results_k100 = load_all_results('text2image-10M', count=100)
        print("Figure 6 (text2image CDF split):")
        plot_split_cdf(all_results['text2image-10M'],
            os.path.join(REPLOT_DIR, 'OverallEval', 'robustness_cdf_split.pdf'),
            results_k100=results_k100 if results_k100 else None)

    elif fig_name == 'fig7':
        print("Figure 7+8 (stacked CDF):")
        plot_stacked_cdf_3(all_results,
            os.path.join(REPLOT_DIR, 'OverallEval', 'robustness_cdf_stacked.pdf'))

    elif fig_name == 'fig9':
        print("Figure 9 (tradeoff):")
        plot_tradeoff_zilliz_scann(all_results['text2image-10M'],
            os.path.join(REPLOT_DIR, 'OverallEval', 'text2image_r30_recall_qps_tradeoff.pdf'))

    elif fig_name == 'fig10':
        print("Figure 10 (robustness-recall):")
        plot_robustness_recall_5panel(all_results['text2image-10M'],
            os.path.join(REPLOT_DIR, 'OverallEval', 'text2image_recall_robustness.pdf'))

    elif fig_name == 'rag':
        print("RAG figure:")
        # Import and run plot_rag
        import subprocess
        subprocess.run([sys.executable, 'plot_rag.py'], check=True)

    else:
        print(f"Unknown figure: {fig_name}")
        print(f"Available: {', '.join(FIGURE_REGISTRY.keys())}")
        return

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('figures', nargs='*', default=['all'],
                       help=f'Figures to generate: {", ".join(FIGURE_REGISTRY.keys())}, or "all"')
    args = parser.parse_args()

    figs = list(FIGURE_REGISTRY.keys()) if 'all' in args.figures else args.figures

    print(f"Generating: {', '.join(figs)}")
    all_results = {}
    for fig in figs:
        print(f"\n{'='*60}")
        all_results = run_figure(fig, all_results)
    print(f"\nDone. Figures in: {REPLOT_DIR}")
