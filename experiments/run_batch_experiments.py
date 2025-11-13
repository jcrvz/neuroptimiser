#!/usr/bin/env python3
"""
Parallel Batch Experiment Runner for Neuromorphic Optimizer v7
================================================================

Runs nengo-neuropti-v7 in parallel across multiple problem configurations:
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

# ==============================================================================
# CONFIGURATION
# ==============================================================================

FUNCTION_IDS = [1, 2, 8, 10, 15, 17, 20, 21, 24]
INSTANCES = list(range(1, 16))  # 1 to 15
DIMENSIONS = [2, 10]

# Parallel execution settings
MAX_WORKERS = min(cpu_count() - 1, 8)  # Leave 1 CPU free, max 8 workers

SCRIPT_PATH = Path(__file__).parent / "nengo-neuropti-v7-batch.py"
RESULTS_DIR = Path(__file__).parent / "batch_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Thread-safe progress tracking
progress_lock = threading.Lock()
progress_state = {
    'completed': 0,
    'skipped': 0,
    'failed': 0,
    'total': 1  # Initialize to 1 to avoid division by zero, will be updated before execution
}

# ==============================================================================
# SETUP
# ==============================================================================

def setup_directories():
    """Create results directory if it doesn't exist"""
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")
    return RESULTS_DIR

def run_single_experiment(args):
    """
    Run a single experiment configuration.

    Args is a tuple: (fid, instance, dims, results_dir)
    This wrapper allows using map() with multiple arguments.
    """
    fid, instance, dims, results_dir = args

    # Construct output filename
    output_file = results_dir / f"result_f{fid:02d}_i{instance:02d}_d{dims:02d}.json"

    # Skip if already exists
    if output_file.exists():
        with progress_lock:
            progress_state['skipped'] += 1
            total_done = (progress_state['completed'] +
                         progress_state['failed'] +
                         progress_state['skipped'])
            progress = 100 * total_done / progress_state['total']
            print(f"⚠️  Skipped f{fid:02d}_i{instance:02d}_d{dims:02d} | "
                  f"Progress: {total_done}/{progress_state['total']} ({progress:.1f}%)")
        return ('skipped', fid, instance, dims, None)

    # Run the experiment (NO TIMEOUT - let it complete naturally)
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             str(fid), str(instance), str(dims), str(output_file)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Load result for immediate verification
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                error = data['performance']['error']
            except:
                error = float('nan')

            with progress_lock:
                progress_state['completed'] += 1
                total_done = (progress_state['completed'] +
                             progress_state['failed'] +
                             progress_state['skipped'])
                progress = 100 * total_done / progress_state['total']
                print(f"✅ f{fid:02d}_i{instance:02d}_d{dims:02d} | "
                      f"Error: {error:.2e} | "
                      f"Progress: {total_done}/{progress_state['total']} ({progress:.1f}%)")

            return ('success', fid, instance, dims, data if 'data' in locals() else None)
        else:
            with progress_lock:
                progress_state['failed'] += 1
                total_done = (progress_state['completed'] +
                             progress_state['failed'] +
                             progress_state['skipped'])
                progress = 100 * total_done / progress_state['total']
                print(f"❌ Failed f{fid:02d}_i{instance:02d}_d{dims:02d} | "
                      f"RC: {result.returncode} | "
                      f"Progress: {total_done}/{progress_state['total']} ({progress:.1f}%)")
                # Print first 200 chars of stderr if available
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}")

            return ('failed', fid, instance, dims, None)

    except Exception as e:
        with progress_lock:
            progress_state['failed'] += 1
            total_done = (progress_state['completed'] +
                         progress_state['failed'] +
                         progress_state['skipped'])
            progress = 100 * total_done / progress_state['total']
            print(f"❌ Error f{fid:02d}_i{instance:02d}_d{dims:02d} | "
                  f"{str(e)[:100]} | "
                  f"Progress: {total_done}/{progress_state['total']} ({progress:.1f}%)")

        return ('error', fid, instance, dims, None)

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
    """Run all experiments in parallel"""

    print("="*70)
    print("PARALLEL BATCH EXPERIMENT RUNNER - Neuromorphic Optimizer v7")
    print("="*70)

    # Setup
    results_dir = setup_directories()

    # Generate all experiment configurations
    experiments = [
        (fid, instance, dims, results_dir)
        for fid in FUNCTION_IDS
        for dims in DIMENSIONS
        for instance in INSTANCES
    ]

    total_experiments = len(experiments)

    # SET TOTAL BEFORE ANY PARALLEL EXECUTION STARTS
    progress_state['total'] = total_experiments

    print(f"\nConfiguration:")
    print(f"  Function IDs: {FUNCTION_IDS}")
    print(f"  Instances: {min(INSTANCES)}-{max(INSTANCES)}")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Parallel workers: {MAX_WORKERS}")
    print(f"  Timeout: None (experiments run to completion)")

    # Check if batch script exists
    if not SCRIPT_PATH.exists():
        print(f"\n❌ Error: Batch script not found: {SCRIPT_PATH}")
        print("Please create nengo-neuropti-v7-batch.py first!")
        return

    print(f"\n{'='*70}")
    print("STARTING PARALLEL EXECUTION")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs
        futures = {executor.submit(run_single_experiment, exp): exp
                  for exp in experiments}

        # Process results as they complete
        for future in as_completed(futures):
            try:
                status, fid, instance, dims, data = future.result()
            except Exception as e:
                exp = futures[future]
                print(f"❌ Exception for {exp}: {e}")

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("BATCH COMPLETION SUMMARY")
    print("="*70)
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {progress_state['completed']}")
    print(f"Skipped (already done): {progress_state['skipped']}")
    print(f"Failed: {progress_state['failed']}")
    print(f"Duration: {duration}")

    if progress_state['completed'] > 0:
        avg_time = duration.total_seconds() / progress_state['completed']
        print(f"Average time per completed experiment: {avg_time:.1f}s")

    # Load all results and create summary
    print("\nLoading all results...")
    all_results = load_all_results(results_dir)

    if all_results:
        df = create_summary_dataframe(all_results)
        print_summary_statistics(df)
        save_summary(df, results_dir, TIMESTAMP)
    else:
        print("No results found to summarize")

    print("\n✅ Parallel batch experiments complete!")
    print(f"   Used {MAX_WORKERS} parallel workers")
    print(f"   Speedup: ~{MAX_WORKERS}x faster than sequential execution")


if __name__ == "__main__":
    main()

