"""Shared constants, palette, and CSV loaders for all figure scripts."""
import csv
from pathlib import Path
from collections import defaultdict

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

# Paul Tol "bright" palette (colorblind-safe)
COLORS = {
    'HNSW':    '#66CCEE',
    'DiskANN': '#4477AA',
    'Zilliz':  '#AA3377',
    'IVFFlat': '#EE6677',
    'ScaNN':   '#FF9D23',
    'Puck':    '#228833',
}

MARKERS = {
    'HNSW':    'o',
    'DiskANN': 's',
    'Zilliz':  'D',
    'IVFFlat': '^',
    'ScaNN':   'v',
    'Puck':    'P',
}

ALGO_ORDER = ['HNSW', 'DiskANN', 'Zilliz', 'IVFFlat', 'ScaNN', 'Puck']

CDF_DELTAS = [i / 10.0 for i in range(11)]  # 0.0, 0.1, ..., 1.0
ROBUSTNESS_DELTAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def load_aggregate(dataset, k=10):
    """Load aggregate metrics CSV. Returns list of dicts matching replot_v2 format."""
    safe = dataset.replace('-', '_')
    path = DATA_DIR / "aggregate" / f"{safe}_k{k}.csv"
    results = []
    with open(path) as f:
        for row in csv.DictReader(f):
            results.append({
                'algo': row['algorithm'],
                'config': row['config'],
                'recall': float(row['recall']),
                'robustness': {d: float(row[f'robustness_{d}']) for d in ROBUSTNESS_DELTAS},
                'qps': float(row['qps']),
            })
    return results


def load_cdf(dataset, k=10):
    """Load CDF CSV. Returns dict {algo: {config, recall, cdf_values}}."""
    safe = dataset.replace('-', '_')
    path = DATA_DIR / "cdf" / f"{safe}_k{k}_cdf.csv"
    result = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            result[row['algorithm']] = {
                'algo': row['algorithm'],
                'config': row['config'],
                'recall': float(row['recall']),
                'cdf_values': [float(row[f'cdf_{d:.1f}']) for d in CDF_DELTAS],
                'cdf_deltas': CDF_DELTAS,
            }
    return result


def load_rag():
    """Load RAG results CSV. Returns {scenario: [(algo, recall, rob, acc), ...]}."""
    path = DATA_DIR / "rag" / "rag_results.csv"
    by_scenario = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            by_scenario[row['scenario']][row['algorithm']].append((
                float(row['recall']),
                float(row['robustness']),
                float(row['accuracy']),
            ))
    return dict(by_scenario)


def find_config_at_recall(results, target=0.9, tolerance=0.02):
    """From aggregate results, select closest config to target recall per algo."""
    by_algo = {}
    for r in results:
        algo = r['algo']
        if abs(r['recall'] - target) > tolerance:
            continue
        if algo not in by_algo or abs(r['recall'] - target) < abs(by_algo[algo]['recall'] - target):
            by_algo[algo] = r
    return by_algo
