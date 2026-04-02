#!/bin/bash
# §4 Metric Comparison — extract metrics and generate r² figure
# Requires: K=10 results for all algorithms on all datasets (run robustness_evaluation.py --run run first)
# Usage: bash run_metric_comparison.sh

set -e

echo "=== Extracting all metrics from HDF5 results ==="
PYTHONPATH="." python extract_all_metrics.py

echo ""
echo "=== Generating r² correlation figure ==="
PYTHONPATH="." python plot_metric_comparison.py

echo ""
echo "=== Copying figure to paper repo ==="
PAPER_DIR="../Robustness-VLDB-Revision"
if [ -d "$PAPER_DIR" ]; then
    cp figures/QueryFeatures/metric_correlation_r2.pdf "$PAPER_DIR/figures/QueryFeatures/metric_correlation_r2.pdf"
    echo "Copied to $PAPER_DIR/figures/QueryFeatures/metric_correlation_r2.pdf"
else
    echo "Paper directory not found at $PAPER_DIR, skipping copy"
fi

echo ""
echo "Done."
