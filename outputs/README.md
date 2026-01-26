# Advanced Metrics & Visualizations

This directory contains scripts for comprehensive performance analysis of AGoT-ReAct vs ReAct approaches.

## Quick Start

Run all analysis with a single command:

```bash
python run_all_analysis.py
```

## Individual Scripts

### 1. Advanced Metrics Analyzer
**File:** `advanced_metrics_analyzer.py`

Computes comprehensive metrics including:
- **Statistical Significance Tests**: T-tests, Chi-square tests, p-values
- **Effect Size Analysis**: Cohen's d
- **Confusion Matrix Metrics**: Accuracy, error rates
- **Reasoning Step Statistics**: Mean, median, quartiles, std deviation
- **Error Analysis**: Patterns in incorrect predictions
- **Efficiency Metrics**: Computational cost comparison
- **Answer Distribution**: Prediction patterns

**Usage:**
```bash
python advanced_metrics_analyzer.py
```

**Output:**
- `gpqa-deepseek/Advanced_Metrics_Report_DeepSeek.txt`
- `gpqa-qwen/Advanced_Metrics_Report_Qwen.txt`

### 2. Visualization Generator
**File:** `visualization_generator.py`

Creates publication-quality charts:
- **Accuracy Comparison**: Bar chart showing accuracy differences
- **Step Distribution**: Histogram of reasoning steps
- **Box Plot**: Statistical distribution of steps
- **Error Analysis**: Stacked bar chart of correct/incorrect answers
- **Efficiency Metrics**: Multi-panel comparison
- **Performance Radar**: Multi-dimensional comparison chart

**Usage:**
```bash
python visualization_generator.py
```

**Output Directories:**
- `gpqa-deepseek/visualizations/`
- `gpqa-qwen/visualizations/`

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- scipy
- matplotlib
- seaborn

## Output Files

### Reports
- Statistical significance analysis
- Detailed performance metrics
- Error pattern analysis
- Efficiency comparisons

### Visualizations (PNG, 300 DPI)
- `{model}_accuracy_comparison.png`
- `{model}_step_distribution.png`
- `{model}_steps_boxplot.png`
- `{model}_error_analysis.png`
- `{model}_efficiency_metrics.png`
- `{model}_performance_radar.png`

## Metrics Explained

### Statistical Significance
- **P-value < 0.05**: Improvement is statistically significant
- **Cohen's d**: Effect size (0.2=small, 0.5=medium, 0.8=large)

### Efficiency Metrics
- **Step Reduction %**: Lower is better (less computation)
- **Accuracy Gain**: Higher is better

## For Your Paper

These scripts generate all figures and statistical tests needed for:
1. Results section
2. Discussion/Analysis section
3. Appendix with detailed metrics

All outputs are publication-ready at 300 DPI.
