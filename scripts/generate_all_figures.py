#!/usr/bin/env python3
"""Generate all paper figures from CSV data."""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

FIGURES = [
    ('fig01_recall_distribution.py', 'Fig 1:  Recall distribution (MSMARCO)'),
    ('fig03_metric_correlation.py',  'Fig 3:  Metric correlation r^2 heatmap'),
    ('fig06_cdf_split.py',           'Fig 6:  CDF split (text2image K=10/K=100)'),
    ('fig07_cdf_stacked.py',         'Fig 7:  CDF stacked (spacev, deep, msmarco)'),
    ('fig08_recall_robustness.py',   'Fig 8:  Recall vs robustness 5-panel'),
    ('fig09_tradeoff.py',            'Fig 9:  Three-way tradeoff'),
    ('fig12_rag.py',                 'Fig 12: RAG results'),
]


def main():
    (SCRIPTS_DIR.parent / 'output').mkdir(exist_ok=True)

    success, failed = 0, 0
    for script, desc in FIGURES:
        print(f"\n{'='*60}")
        print(f"Generating {desc}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script)],
            cwd=str(SCRIPTS_DIR),
        )
        if result.returncode == 0:
            success += 1
        else:
            failed += 1
            print(f"  FAILED (exit code {result.returncode})")

    print(f"\n{'='*60}")
    print(f"Done: {success} succeeded, {failed} failed")
    print(f"Figures saved to: {SCRIPTS_DIR.parent / 'output'}/")


if __name__ == '__main__':
    main()
