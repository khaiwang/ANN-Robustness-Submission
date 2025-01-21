# Robustness Analysis Features

This document describes the new robustness analysis features added to the big-ann-benchmarks plotting system.

## Overview

The plotting system now supports three main types of visualizations:
1. **Normal plots**: Standard 2D plots comparing two metrics
2. **CDF plots**: Robustness analysis using cumulative distribution functions
3. **Filter plots**: Algorithm filtering based on metric constraints

## Robustness Metric Format

The robustness metric naming includes threshold:

- **Format**: `robustness-{threshold}`
- **Example**: `robustness-0.5` means robustness at 50% recall@k threshold
- **Parameters**:
  - `threshold`: Recall threshold (0.0 to 1.0)

## New Plot Types

### 1. CDF Plots (Robustness Analysis)

**Purpose**: Analyze algorithm robustness across different recall thresholds using cumulative distribution functions.

**Usage**:
```bash
python plot.py -T cdf --fix-recall 95 --dataset <dataset> --count <k>
```

**Parameters**:
- `-T cdf`: Specifies CDF plot type
- `--fix-recall <value>`: Target recall rate (0-100, default: 90)

**What it does**:
1. Finds the best configuration for each algorithm that achieves the target recall rate
2. Computes CDF data showing how each algorithm performs across different recall thresholds
3. Plots robustness curves showing the frequency of queries achieving various recall levels

**Output**: A plot showing:
- X-axis: Recall rate thresholds (0.0 to 1.0)
- Y-axis: Robustness value (frequency of queries achieving ≥ threshold recall)
- Multiple curves, one per algorithm

### 2. Filter Plots (Metric-Based Filtering)

**Purpose**: Compare algorithms while filtering to only show configurations within specific metric bounds.

**Usage**:
```bash
python plot.py -T filter --fix-metric k-nn --min 0.8 --max 0.95 --dataset <dataset> --count <k>
```

**Parameters**:
- `-T filter`: Specifies filter plot type
- `--fix-metric <metric>`: Metric to filter on (e.g., "k-nn", "robustness-0.5@10")
- `--min <value>`: Minimum value for the fixed metric (default: 0)
- `--max <value>`: Maximum value for the fixed metric (default: ∞)

**What it does**:
1. Computes the specified metric for all algorithm configurations
2. Filters to only include configurations where the metric falls within the specified range
3. Creates a standard 2D plot using only the filtered configurations

**Output**: A standard 2D plot with title indicating the filtering constraints.

## Examples

### Example 1: Robustness CDF Analysis
```bash
# Generate robustness CDF plot with 95% target recall
python plot.py -T cdf \
    --fix-recall 95 \
    --dataset text2image-10M \
    --count 10 \
    --x-axis k-nn \
    --y-axis qps
```

### Example 2: Filter by Recall Range
```bash
# Compare algorithms only for configurations achieving 80-95% recall
python plot.py -T filter \
    --fix-metric k-nn \
    --min 0.8 \
    --max 0.95 \
    --dataset text2image-10M \
    --count 10 \
    --x-axis k-nn \
    -X a4 \
    --y-axis qps
```

## Technical Details

### CDF Computation
The CDF plotting system:
1. Uses `compute_cdf_to_robustness_values()` to generate CDF data
2. Finds optimal configurations using `find_configuration_with_fixed_recall()`
3. Plots cumulative distribution functions showing robustness across thresholds

### Filtering Logic
The filtering system:
1. Uses `compute_metrics()` to calculate the specified metric
2. Applies `find_configuration_with_fixed_range()` for filtering
3. Creates standard plots with only filtered configurations

## File Changes

### Modified Files:
- `plot.py`: Added CDF and filter plotting functions
- `benchmark/plotting/metrics.py`: Added robustness metric

### New Functions:
- `create_plot_cdf()`: Generates robustness CDF plots
- `create_plot_with_fixed_metric()`: Generates filtered plots
- `robustness_metric()`: Supports robustness metric

## Usage Tips

1. **CDF Plots**: Use for comprehensive robustness analysis across all recall thresholds
2. **Filter Plots**: Use when you want to focus on specific performance ranges
3. **Metric Selection**: Choose metrics that are meaningful for your analysis (e.g., k-nn for recall, qps for speed)
4. **Threshold Values**: For robustness, typical thresholds are 0.1, 0.3, 0.5, 0.7, 0.9







feat: Add robustness CDF plots and metric filtering

- Add Robustness metric (robustness-x@k)
- Add CDF plots for robustness analysis across recall thresholds
- Add filter plots for metric-based algorithm comparison  
- Update robustness metric format to robustness-x@k

Usage: python plot.py -T cdf|filter --fix-recall|--fix-metric ...