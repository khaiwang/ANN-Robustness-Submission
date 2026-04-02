"""
§5.4 Index Family Analysis
- GT rank distribution: which GT ranks each algorithm retrieves in severe failures
- K=100 retrieve-and-rerank: does larger K recover GT#1 for HNSW severe failures

Requires: K=10 results for all algorithms, K=100 results for faiss_hnsw on text2image
Usage: PYTHONPATH="." python analyze_index_families.py
"""
import h5py
import numpy as np
import glob
import os
from benchmark.datasets import DATASETS


GRAPH_FAMILY = {'faiss_hnsw', 'diskann', 'zilliz'}
PARTITION_FAMILY = {'scann', 'puck', 'faiss-ivf', 'faiss-ivfpqfs'}


def load_gt(dataset, k=100):
    ds = DATASETS[dataset]()
    gt_ids, _ = ds.get_groundtruth(k)
    return gt_ids


def find_config_at_recall(results_dir, gt_top10, target=0.87):
    """Find the config closest to target recall."""
    nq = gt_top10.shape[0]
    best = None
    for fp in sorted(glob.glob(f'{results_dir}/*.hdf5')):
        try:
            with h5py.File(fp, 'r') as f:
                nn = np.array(f['neighbors'])
            recall = np.mean([
                len(set(int(x) for x in nn[i][:10]) & set(int(x) for x in gt_top10[i])) / 10
                for i in range(nq)
            ])
            if best is None or abs(recall - target) < abs(best[1] - target):
                best = (fp, recall, nn)
        except Exception:
            pass
    return best


def gt_rank_analysis(datasets=None, target_recall=0.87):
    """Analyze GT rank distribution of retrieved vectors in severe failures."""
    if datasets is None:
        datasets = ['text2image-10M', 'msspacev-10M', 'deep-10M', 'msmarco-10M']

    bins = [(0, 10, 'GT 1-10'), (10, 20, 'GT 11-20'), (20, 50, 'GT 21-50'),
            (50, 100, 'GT 51-100'), (100, 10000, 'Outside')]

    algos_per_dataset = {
        'text2image-10M': ['faiss_hnsw', 'diskann', 'zilliz', 'scann', 'puck', 'faiss-ivfpqfs', 'faiss-ivf'],
        'msspacev-10M':   ['faiss_hnsw', 'diskann', 'zilliz', 'scann', 'puck', 'faiss-ivfpqfs', 'faiss-ivf'],
        'deep-10M':       ['faiss_hnsw', 'diskann', 'zilliz', 'scann', 'puck', 'faiss-ivfpqfs', 'faiss-ivf'],
        'msmarco-10M':    ['faiss_hnsw', 'diskann', 'scann', 'puck', 'faiss-ivf'],
    }

    for dataset in datasets:
        gt_ids = load_gt(dataset)
        nq = gt_ids.shape[0]
        gt10 = gt_ids[:, :10]
        results_base = f'results/neurips23/ood/{dataset}/10'

        print(f'\n{"=" * 70}')
        print(f'  {dataset} (nq={nq})')
        print(f'{"=" * 70}')

        for algo_dir in algos_per_dataset.get(dataset, []):
            pick = find_config_at_recall(f'{results_base}/{algo_dir}', gt10, target_recall)
            if pick is None:
                continue

            fp, recall_val, nn = pick
            severe_ranks = []
            n_severe = 0
            n_total_fail = 0

            for i in range(nq):
                retrieved = set(int(x) for x in nn[i][:10])
                per_q_recall = len(retrieved & set(int(x) for x in gt10[i])) / 10
                if per_q_recall > 0.3:
                    continue
                n_severe += 1
                if per_q_recall == 0:
                    n_total_fail += 1
                gt_rank_map = {int(gt_ids[i][r]): r for r in range(100)}
                for v in nn[i][:10]:
                    severe_ranks.append(gt_rank_map.get(int(v), 500))

            family = 'G' if algo_dir in GRAPH_FAMILY else 'P'
            r = np.array(severe_ranks) if severe_ranks else np.array([])
            total = len(r)

            print(f'  {algo_dir:15s} [{family}] recall={recall_val:.3f} | severe={n_severe:5d} (r=0:{n_total_fail:4d}) | ', end='')
            if total > 0:
                for lo, hi, blabel in bins:
                    count = np.sum((r >= lo) & (r < hi))
                    pct = count / total * 100
                    print(f'{blabel}:{pct:5.1f}%  ', end='')
            print()


def rerank_analysis(dataset='text2image-10M'):
    """Analyze whether K=100 recovers GT#1 for HNSW severe K=10 failures."""
    gt_ids = load_gt(dataset)
    nq = gt_ids.shape[0]
    gt_top10 = gt_ids[:, :10]

    # K=10 baseline at ef=128, M=32
    k10_path = 'results/neurips23/ood/text2image-10M/10/faiss_hnsw/angular_M_32_efConstruction_500_ef_128.hdf5'
    if not os.path.exists(k10_path):
        print(f'K=10 baseline not found: {k10_path}')
        return

    with h5py.File(k10_path, 'r') as f:
        nn_k10 = np.array(f['neighbors'])

    # Identify severe K=10 failures
    severe_queries = []
    for i in range(nq):
        ret = set(int(x) for x in nn_k10[i][:10])
        if len(ret & set(int(x) for x in gt_top10[i])) / 10 <= 0.3:
            severe_queries.append(i)

    print(f'\n{"=" * 70}')
    print(f'  K=100 Retrieve-and-Rerank Analysis ({dataset})')
    print(f'  Severe K=10 failures (ef=128, M=32): {len(severe_queries)}')
    print(f'{"=" * 70}')
    print(f'{"Config":>55s} |  ef  | recall@100 |  GT#1 | GT1-5 | All10')

    k100_dir = f'results/neurips23/ood/{dataset}/100/faiss_hnsw'
    results = []
    for fp in sorted(glob.glob(f'{k100_dir}/*.hdf5')):
        with h5py.File(fp, 'r') as f:
            nn100 = np.array(f['neighbors'])

        config = os.path.basename(fp).replace('.hdf5', '')
        ef = int(config.split('_ef_')[1]) if '_ef_' in config else 0

        recall100 = np.mean([
            len(set(int(x) for x in nn100[i]) & set(int(x) for x in gt_ids[i])) / 100
            for i in range(nq)
        ])

        gt1 = gt5 = all10 = 0
        for i in severe_queries:
            ret = set(int(x) for x in nn100[i])
            if int(gt_ids[i][0]) in ret:
                gt1 += 1
            if any(int(gt_ids[i][r]) in ret for r in range(5)):
                gt5 += 1
            if all(int(gt_ids[i][r]) in ret for r in range(10)):
                all10 += 1

        n = len(severe_queries)
        results.append((ef, config, recall100, gt1 / n, gt5 / n, all10 / n))

    results.sort()
    for ef, config, r100, g1, g5, a10 in results:
        marker = ''
        if ef == 128:
            marker = ' <-- ef=K'
        elif ef == 256:
            marker = ' <-- 2x K'
        elif ef == 512:
            marker = ' <-- typical'
        print(f'{config:>55s} | {ef:4d} | {r100:10.4f} | {g1:4.1f}% | {g5:4.1f}% | {a10:4.1f}%{marker}')


if __name__ == '__main__':
    print('=== GT Rank Distribution Analysis (§5.4) ===')
    gt_rank_analysis()

    print('\n\n=== K=100 Retrieve-and-Rerank Analysis (§5.4) ===')
    rerank_analysis()
