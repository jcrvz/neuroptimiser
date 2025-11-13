# Batch Experiments - Neuromorphic Optimizer v7

This directory contains scripts for running batch experiments with the neuromorphic optimizer.

## Files

- **`nengo-neuropti-v7-batch.py`**: Modified version of the main optimizer that accepts command-line arguments and saves results to JSON (no plotting)
- **`run_batch_experiments.py`**: Batch runner that executes experiments across all configurations

## Quick Start

Run all experiments:

```bash
python run_batch_experiments.py
```

This will:
- Run experiments for all combinations of:
  - Function IDs: 1, 2, 8, 10, 15, 17, 20, 21, 24
  - Instances: 1-15
  - Dimensions: 2, 10
- Total: 9 functions × 15 instances × 2 dimensions = **270 experiments**
- Save individual results to `batch_results/result_f{fid:02d}_i{instance:02d}_d{dims:02d}.json`
- Generate summary CSV and statistics JSON

## Manual Single Experiment

Run a single experiment manually:

```bash
python nengo-neuropti-v7-batch.py <fid> <instance> <dims> <output_file>
```

Example:
```bash
python nengo-neuropti-v7-batch.py 1 1 10 my_result.json
```

## Results Structure

### Individual Result JSON

Each experiment saves a JSON file with:

```json
{
  "configuration": {
    "simulation_time": 20.0,
    "lambda": 50,
    "mu": 25,
    ...
  },
  "problem": {
    "function_id": 1,
    "instance": 1,
    "dimension": 10,
    ...
  },
  "performance": {
    "total_evaluations": 1000000,
    "best_fitness": 0.123456,
    "optimal_fitness": 0.0,
    "error": 0.123456
  },
  "operator_usage": {
    "LF": 150,
    "DM": 200,
    "PS": 300,
    "SP": 350
  },
  "operator_percentages": {
    "LF": 15.0,
    "DM": 20.0,
    "PS": 30.0,
    "SP": 35.0
  },
  "operator_weights": {
    "LF": 1.234,
    "DM": 0.987,
    "PS": 2.456,
    "SP": 1.789
  },
  "best_solution": {
    "v_space": [...],
    "x_space": [...]
  },
  "optimal_solution": {
    "x_space": [...]
  }
}
```

### Summary CSV

The batch runner generates a summary CSV with columns:
- `function_id`, `instance`, `dimension`
- `total_evaluations`, `best_fitness`, `optimal_fitness`, `error`, `relative_error`
- `simulation_time`
- `op_LF_count`, `op_LF_percent`, `op_LF_weight`
- `op_DM_count`, `op_DM_percent`, `op_DM_weight`
- `op_PS_count`, `op_PS_percent`, `op_PS_weight`
- `op_SP_count`, `op_SP_percent`, `op_SP_weight`

### Statistics JSON

Aggregated statistics including:
- Overall error statistics (mean, median, std, min, max)
- Breakdown by dimension
- Breakdown by function
- Average operator usage percentages

## Resuming Interrupted Runs

The batch runner automatically skips experiments that already have result files, so you can safely interrupt and resume.

## Expected Runtime

- Single experiment: ~20-30 seconds
- Full batch (270 experiments): ~2-3 hours

## Output Directory

All results are saved to:
```
batch_results/
├── result_f01_i01_d02.json
├── result_f01_i01_d10.json
├── result_f01_i02_d02.json
├── ...
├── summary_YYYYMMDD_HHMMSS.csv
└── statistics_YYYYMMDD_HHMMSS.json
```

## Analysis

After experiments complete, you can analyze the results using pandas:

```python
import pandas as pd

# Load summary
df = pd.read_csv('batch_results/summary_YYYYMMDD_HHMMSS.csv')

# Group by dimension
print(df.groupby('dimension')['error'].describe())

# Group by function
print(df.groupby('function_id')['error'].describe())

# Operator usage correlation with performance
print(df[['error', 'op_LF_percent', 'op_DM_percent', 'op_PS_percent', 'op_SP_percent']].corr())
```

## Troubleshooting

**Problem**: Import errors
- **Solution**: Make sure you're in the correct conda/virtual environment with nengo, ioh, etc. installed

**Problem**: Timeout errors
- **Solution**: Increase timeout in `run_batch_experiments.py` (line ~60): `timeout=600` for 10 minutes

**Problem**: Memory issues
- **Solution**: The experiments run sequentially to avoid memory issues. If still problematic, reduce `LAMBDA` or `SIMULATION_TIME` in the batch script.

## Citation

If you use these experiments, please cite:
- Nengo: Bekolay et al. (2014). Frontiers in Neuroinformatics, 7, 48.
- IOHexperimenter: Doerr et al. (2020). arXiv:2007.03953

