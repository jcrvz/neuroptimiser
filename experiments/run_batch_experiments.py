#!/usr/bin/env python3
"""
Batch Experiment Runner for Neuromorphic Optimizer v7
======================================================

Runs nengo-neuropti-v7 across multiple problem configurations:
- Function IDs: 1, 2, 8, 10, 15, 17, 20, 21, 24
- Instances: 1-15
- Dimensions: 2, 10

Results are saved to JSON and CSV files for later analysis.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================

FUNCTION_IDS = [1, 2, 8, 10, 15, 17, 20, 21, 24]
INSTANCES = list(range(1, 16))  # 1 to 15
DIMENSIONS = [2, 5, 10]

SCRIPT_PATH = Path(__file__).parent / "nengo-neuropti-v7-batch.py"
RESULTS_DIR = Path(__file__).parent / "batch_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ==============================================================================
# SETUP
# ==============================================================================

def setup_directories():
    """Create results directory if it doesn't exist"""
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")
    return RESULTS_DIR

def run_single_experiment(fid, instance, dims, results_dir):
    """Run a single experiment configuration"""

    print(f"\n{'='*70}")
    print(f"Running: FID={fid}, Instance={instance}, Dims={dims}")
    print(f"{'='*70}")

    # Construct output filename
    output_file = results_dir / f"result_f{fid:02d}_i{instance:02d}_d{dims:02d}.json"

    # Skip if already exists
    if output_file.exists():
        print(f"⚠️  Result already exists, skipping: {output_file.name}")
        return None

    # Run the experiment
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             str(fid), str(instance), str(dims), str(output_file)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout per experiment
        )

        if result.returncode == 0:
            print(f"✅ Success: {output_file.name}")

            # Load and return the result for immediate summary
            with open(output_file, 'r') as f:
                data = json.load(f)
            return data
        else:
            print(f"❌ Failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout expired for this experiment")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def load_all_results(results_dir):
    """Load all JSON results from the results directory"""
    results = []

    for json_file in sorted(results_dir.glob("result_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results

def create_summary_dataframe(results):
    """Convert results list to pandas DataFrame"""

    if not results:
        return None

    rows = []
    for r in results:
        row = {
            'function_id': r['problem']['function_id'],
            'instance': r['problem']['instance'],
            'dimension': r['problem']['dimension'],
            'total_evaluations': r['performance']['total_evaluations'],
            'best_fitness': r['performance']['best_fitness'],
            'optimal_fitness': r['performance']['optimal_fitness'],
            'error': r['performance']['error'],
            'relative_error': r['performance']['error'] / (abs(r['performance']['optimal_fitness']) + 1e-12),
            'simulation_time': r['configuration']['simulation_time'],
        }

        # Add operator usage
        for op_name, count in r['operator_usage'].items():
            row[f'op_{op_name}_count'] = count
            row[f'op_{op_name}_percent'] = r['operator_percentages'][op_name]
            row[f'op_{op_name}_weight'] = r['operator_weights'][op_name]

        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def print_summary_statistics(df):
    """Print summary statistics"""

    if df is None or len(df) == 0:
        print("No results to summarize")
        return

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print(f"\nTotal experiments completed: {len(df)}")
    print(f"Total evaluations: {df['total_evaluations'].sum():,}")

    print("\n--- Error Statistics ---")
    print(f"Mean error: {df['error'].mean():.6e}")
    print(f"Median error: {df['error'].median():.6e}")
    print(f"Min error: {df['error'].min():.6e}")
    print(f"Max error: {df['error'].max():.6e}")
    print(f"Std error: {df['error'].std():.6e}")

    print("\n--- By Dimension ---")
    for dim in sorted(df['dimension'].unique()):
        df_dim = df[df['dimension'] == dim]
        print(f"D={dim}: {len(df_dim)} experiments, "
              f"mean error = {df_dim['error'].mean():.6e}, "
              f"median error = {df_dim['error'].median():.6e}")

    print("\n--- By Function ID ---")
    for fid in sorted(df['function_id'].unique()):
        df_fid = df[df['function_id'] == fid]
        print(f"F{fid:02d}: {len(df_fid)} experiments, "
              f"mean error = {df_fid['error'].mean():.6e}, "
              f"median error = {df_fid['error'].median():.6e}")

    print("\n--- Operator Usage (Average %) ---")
    op_cols = [col for col in df.columns if col.startswith('op_') and col.endswith('_percent')]
    for col in sorted(op_cols):
        op_name = col.replace('op_', '').replace('_percent', '')
        print(f"{op_name.upper():4s}: {df[col].mean():6.2f}% ± {df[col].std():5.2f}%")

def save_summary(df, results_dir, timestamp):
    """Save summary CSV and statistics"""

    if df is None or len(df) == 0:
        print("No results to save")
        return

    # Save full results
    csv_file = results_dir / f"summary_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Summary saved to: {csv_file}")

    # Save aggregated statistics
    stats = {
        'total_experiments': len(df),
        'total_evaluations': int(df['total_evaluations'].sum()),
        'timestamp': timestamp,
        'error_statistics': {
            'mean': float(df['error'].mean()),
            'median': float(df['error'].median()),
            'std': float(df['error'].std()),
            'min': float(df['error'].min()),
            'max': float(df['error'].max()),
        },
        'by_dimension': {},
        'by_function': {},
        'operator_usage_avg': {}
    }

    # By dimension
    for dim in sorted(df['dimension'].unique()):
        df_dim = df[df['dimension'] == dim]
        stats['by_dimension'][int(dim)] = {
            'count': len(df_dim),
            'mean_error': float(df_dim['error'].mean()),
            'median_error': float(df_dim['error'].median()),
        }

    # By function
    for fid in sorted(df['function_id'].unique()):
        df_fid = df[df['function_id'] == fid]
        stats['by_function'][int(fid)] = {
            'count': len(df_fid),
            'mean_error': float(df_fid['error'].mean()),
            'median_error': float(df_fid['error'].median()),
        }

    # Operator usage
    for op_name in ['LF', 'DM', 'PS', 'SP']:
        col = f'op_{op_name}_percent'
        if col in df.columns:
            stats['operator_usage_avg'][op_name] = {
                'mean_percent': float(df[col].mean()),
                'std_percent': float(df[col].std()),
            }

    stats_file = results_dir / f"statistics_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Statistics saved to: {stats_file}")

# ==============================================================================
# MAIN BATCH RUNNER
# ==============================================================================

def main():
    """Run all experiments"""

    print("="*70)
    print("BATCH EXPERIMENT RUNNER - Neuromorphic Optimizer v7")
    print("="*70)

    # Setup
    results_dir = setup_directories()

    # Calculate total experiments
    total_experiments = len(FUNCTION_IDS) * len(INSTANCES) * len(DIMENSIONS)
    print(f"\nConfiguration:")
    print(f"  Function IDs: {FUNCTION_IDS}")
    print(f"  Instances: {min(INSTANCES)}-{max(INSTANCES)}")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Total experiments: {total_experiments}")

    # Check if batch script exists
    if not SCRIPT_PATH.exists():
        print(f"\n❌ Error: Batch script not found: {SCRIPT_PATH}")
        print("Creating it now...")
        create_batch_script()

    # Run all experiments
    completed = 0
    failed = 0
    skipped = 0

    start_time = datetime.now()

    for fid in FUNCTION_IDS:
        for dims in DIMENSIONS:
            for instance in INSTANCES:
                result = run_single_experiment(fid, instance, dims, results_dir)

                if result is not None:
                    completed += 1
                elif (results_dir / f"result_f{fid:02d}_i{instance:02d}_d{dims:02d}.json").exists():
                    skipped += 1
                else:
                    failed += 1

                # Print progress
                total_done = completed + failed + skipped
                progress = 100 * total_done / total_experiments
                print(f"Progress: {total_done}/{total_experiments} ({progress:.1f}%) - "
                      f"✅ {completed} | ⚠️ {skipped} | ❌ {failed}")

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("BATCH COMPLETION SUMMARY")
    print("="*70)
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    print(f"Duration: {duration}")

    # Load all results and create summary
    print("\nLoading all results...")
    all_results = load_all_results(results_dir)

    if all_results:
        df = create_summary_dataframe(all_results)
        print_summary_statistics(df)
        save_summary(df, results_dir, TIMESTAMP)
    else:
        print("No results found to summarize")

    print("\n✅ Batch experiments complete!")

def create_batch_script():
    """Create the batch-compatible version of the main script"""
    # This will be called if the batch script doesn't exist
    # We'll read the original and modify it
    pass

if __name__ == "__main__":
    main()

