#!/usr/bin/env python3
"""
Analyze Batch Experiment Results
=================================

This script loads and analyzes the batch experiment results,
generating summary statistics and visualizations.

Usage:
    python analyze_results.py [results_dir]

If no directory is specified, uses 'batch_results/'
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

if len(sys.argv) > 1:
    RESULTS_DIR = Path(sys.argv[1])
else:
    RESULTS_DIR = Path(__file__).parent / "batch_results"

# ==============================================================================
# LOAD RESULTS
# ==============================================================================

def load_all_results(results_dir):
    """Load all JSON results"""
    results = []

    json_files = sorted(results_dir.glob("result_*.json"))

    if not json_files:
        print(f"No result files found in {results_dir}")
        return []

    print(f"Loading {len(json_files)} result files...")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results

def create_dataframe(results):
    """Convert results to DataFrame"""

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
            'abs_error': abs(r['performance']['error']),
            'log_error': np.log10(abs(r['performance']['error']) + 1e-16),
        }

        # Add operator usage
        for op_name in ['LF', 'DM', 'PS', 'SP']:
            row[f'op_{op_name}_count'] = r['operator_usage'].get(op_name, 0)
            row[f'op_{op_name}_percent'] = r['operator_percentages'].get(op_name, 0.0)
            row[f'op_{op_name}_weight'] = r['operator_weights'].get(op_name, 1.0)

        rows.append(row)

    return pd.DataFrame(rows)

# ==============================================================================
# ANALYSIS
# ==============================================================================

def create_function_dimension_statistics(df):
    """
    Create detailed statistics DataFrame for each function-dimension combination.

    Returns a DataFrame with rows for each (function_id, dimension) pair and
    columns for various statistics on best_fitness and error.
    """

    if df is None or len(df) == 0:
        return None

    stats_rows = []

    # Iterate over all combinations of function_id and dimension
    for fid in sorted(df['function_id'].unique()):
        for dim in sorted(df['dimension'].unique()):
            # Filter data for this combination
            df_subset = df[(df['function_id'] == fid) & (df['dimension'] == dim)]

            if len(df_subset) == 0:
                continue

            # Calculate statistics
            stats = {
                'function_id': fid,
                'dimension': dim,
                'n_instances': len(df_subset),

                # Best fitness statistics
                'best_fitness_mean': df_subset['best_fitness'].mean(),
                'best_fitness_median': df_subset['best_fitness'].median(),
                'best_fitness_std': df_subset['best_fitness'].std(),
                'best_fitness_min': df_subset['best_fitness'].min(),
                'best_fitness_max': df_subset['best_fitness'].max(),
                'best_fitness_q25': df_subset['best_fitness'].quantile(0.25),
                'best_fitness_q75': df_subset['best_fitness'].quantile(0.75),

                # Error statistics (absolute)
                'error_mean': df_subset['abs_error'].mean(),
                'error_median': df_subset['abs_error'].median(),
                'error_std': df_subset['abs_error'].std(),
                'error_min': df_subset['abs_error'].min(),
                'error_max': df_subset['abs_error'].max(),
                'error_q25': df_subset['abs_error'].quantile(0.25),
                'error_q75': df_subset['abs_error'].quantile(0.75),

                # Log error statistics
                'log_error_mean': df_subset['log_error'].mean(),
                'log_error_median': df_subset['log_error'].median(),
                'log_error_std': df_subset['log_error'].std(),

                # Success metrics
                'success_rate_1e-4': (df_subset['abs_error'] < 1e-4).sum() / len(df_subset) * 100,
                'success_rate_1e-6': (df_subset['abs_error'] < 1e-6).sum() / len(df_subset) * 100,
                'success_rate_1e-8': (df_subset['abs_error'] < 1e-8).sum() / len(df_subset) * 100,

                # Optimal fitness (should be constant for all instances)
                'optimal_fitness': df_subset['optimal_fitness'].iloc[0],
            }

            stats_rows.append(stats)

    # Create DataFrame
    stats_df = pd.DataFrame(stats_rows)

    return stats_df

def print_summary(df):
    """Print summary statistics"""

    print("\n" + "="*70)
    print("BATCH RESULTS SUMMARY")
    print("="*70)

    print(f"\nTotal experiments: {len(df)}")
    print(f"Total evaluations: {df['total_evaluations'].sum():,}")

    print("\n--- Error Statistics (Absolute) ---")
    print(f"Mean:   {df['abs_error'].mean():.6e}")
    print(f"Median: {df['abs_error'].median():.6e}")
    print(f"Std:    {df['abs_error'].std():.6e}")
    print(f"Min:    {df['abs_error'].min():.6e}")
    print(f"Max:    {df['abs_error'].max():.6e}")

    print("\n--- Error by Dimension ---")
    for dim in sorted(df['dimension'].unique()):
        df_dim = df[df['dimension'] == dim]
        print(f"\n{dim}D ({len(df_dim)} experiments):")
        print(f"  Mean error:   {df_dim['abs_error'].mean():.6e}")
        print(f"  Median error: {df_dim['abs_error'].median():.6e}")
        print(f"  Success rate (error < 1e-4): {100 * (df_dim['abs_error'] < 1e-4).sum() / len(df_dim):.1f}%")

    print("\n--- Error by Function ---")
    summary_by_func = df.groupby('function_id').agg({
        'abs_error': ['count', 'mean', 'median', 'std', 'min', 'max']
    }).round(6)
    print(summary_by_func)

    print("\n--- Operator Usage (Average %) ---")
    for op in ['LF', 'DM', 'PS', 'SP']:
        col = f'op_{op}_percent'
        print(f"{op}: {df[col].mean():6.2f}% ± {df[col].std():5.2f}%")

    print("\n--- Best Performers (Top 10 by error) ---")
    top10 = df.nsmallest(10, 'abs_error')[['function_id', 'instance', 'dimension', 'abs_error']]
    print(top10.to_string(index=False))

    print("\n--- Worst Performers (Bottom 10 by error) ---")
    bottom10 = df.nlargest(10, 'abs_error')[['function_id', 'instance', 'dimension', 'abs_error']]
    print(bottom10.to_string(index=False))

def plot_results(df, output_dir):
    """Generate publication-quality analysis plots"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Check LaTeX availability
    use_latex = False
    try:
        import matplotlib
        matplotlib.rcParams['text.usetex'] = True
        fig_test = plt.figure()
        plt.close(fig_test)
        use_latex = True
        print("✅ LaTeX rendering enabled")
    except:
        print("⚠️  LaTeX not available - using standard fonts")
        use_latex = False

    # Publication-quality style settings
    fontsize = 14
    plt.rcParams.update({
        # LaTeX rendering (conditional)
        'text.usetex': use_latex,

        # Fonts
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'] if use_latex else ['DejaVu Serif'],
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize,

        # Figure appearance
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Lines and markers
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.linewidth': 0.8,

        # Grid
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.color': '0.7',
        'axes.grid': False,
        'axes.axisbelow': True,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,

        # Spines
        'axes.spines.top': True,
        'axes.spines.right': True,
    })

    # Add LaTeX preamble if available
    if use_latex:
        try:
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        except:
            pass

    # Color palette for operators (colorblind-friendly)
    op_colors = {
        'LF': '#E69F00',  # Orange
        'DM': '#56B4E9',  # Sky blue
        'PS': '#009E73',  # Bluish green
        'SP': '#D55E00',  # Vermillion
    }

    # 1. Error distribution by dimension (violin plot)
    fig, ax = plt.subplots(figsize=(7, 5))

    dims = sorted(df['dimension'].unique())
    positions = np.arange(len(dims))

    parts = ax.violinplot(
        [df[df['dimension'] == d]['log_error'].values for d in dims],
        positions=positions,
        widths=0.6,
        showmeans=True,
        showextrema=True,
        showmedians=True
    )

    # Customize violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('#56B4E9')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels([f'${d}$D' for d in dims])
    ax.set_xlabel('Problem Dimension')
    ax.set_ylabel(r'$\log_{10}(\mathrm{Error})$')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution_by_dimension.pdf', dpi=300, transparent=True)
    plt.savefig(output_dir / 'error_distribution_by_dimension.png', dpi=300, transparent=False)
    print(f"Saved: {output_dir / 'error_distribution_by_dimension.pdf'}")
    plt.close()

    # 2. Error by function (grouped bar chart with error bars)
    fig, ax = plt.subplots(figsize=(10, 5))

    function_ids = sorted(df['function_id'].unique())
    x = np.arange(len(function_ids))
    width = 0.35

    dims = sorted(df['dimension'].unique())

    for idx, dim in enumerate(dims):
        df_dim = df[df['dimension'] == dim]
        means = []
        stds = []

        for fid in function_ids:
            df_func = df_dim[df_dim['function_id'] == fid]
            means.append(df_func['log_error'].mean())
            stds.append(df_func['log_error'].std())

        offset = width * (idx - 0.5)
        bars = ax.bar(x + offset, means, width, yerr=stds,
                     label=f'${dim}$D', alpha=0.8, capsize=3,
                     color=['#56B4E9', '#84E690', '#E69F00'][idx])

    ax.set_xlabel('Function ID')
    ax.set_ylabel(r'Mean $\log_{10}(\mathrm{Error})$')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{fid}' for fid in function_ids])
    ax.legend(loc='upper left', ncol=3)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_by_function.pdf', dpi=300, transparent=True)
    plt.savefig(output_dir / 'error_by_function.png', dpi=300, transparent=False)
    print(f"Saved: {output_dir / 'error_by_function.pdf'}")
    plt.close()

    # 2. Error by function (violin plots grouped by dimension)
    fig, ax = plt.subplots(figsize=(12, 5))

    function_ids = sorted(df['function_id'].unique())
    dims = sorted(df['dimension'].unique())

    # Calculate positions for grouped violins
    n_dims = len(dims)
    width = 0.8 / n_dims
    positions_base = np.arange(len(function_ids))

    # Color palette for dimensions
    dim_colors = ['#56B4E9', '#84E690', '#E69F00'][:n_dims]

    for dim_idx, dim in enumerate(dims):
        df_dim = df[df['dimension'] == dim]

        # Prepare data for each function
        violin_data = []
        positions = []

        for func_idx, fid in enumerate(function_ids):
            df_func = df_dim[df_dim['function_id'] == fid]
            if len(df_func) > 0:
                violin_data.append(df_func['log_error'].values)
                positions.append(positions_base[func_idx] + width * (dim_idx - n_dims / 2 + 0.5))

        # Create violin plot
        parts = ax.violinplot(
            violin_data,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            showextrema=True,
            showmedians=True
        )

        # Customize violins
        for pc in parts['bodies']:
            pc.set_facecolor(dim_colors[dim_idx])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)

        if 'cmedians' in parts:
            parts['cmedians'].set_edgecolor('red')
            parts['cmedians'].set_linewidth(1.2)

        if 'cmaxs' in parts:
            parts['cmaxs'].set_edgecolor('black')
            parts['cmaxs'].set_linewidth(1.2)

        if 'cmins' in parts:
            parts['cmins'].set_edgecolor('black')
            parts['cmins'].set_linewidth(1.2)

        if 'cmeans' in parts:
            parts['cmeans'].set_edgecolor('blue')
            parts['cmeans'].set_linewidth(1.2)

        if 'cbars' in parts:
            parts['cbars'].set_edgecolor('black')
            parts['cbars'].set_linewidth(1.2)

        # Add to legend (use first violin as proxy)
        if violin_data:
            ax.plot([], [], color=dim_colors[dim_idx], linewidth=8,
                    alpha=0.7, label=f'${dim}$D')

        # Add quartile markers
        for i, fid in enumerate(function_ids):
            data = df[df['function_id'] == fid]['log_error'].values
            q1, q3 = np.percentile(data, [25, 75])
            ax.plot([i, i], [q1, q3], color='black', linewidth=2, alpha=0.8)

    ax.set_xlabel('Function ID')
    ax.set_ylabel(r'$\log_{10}(\mathrm{Error})$')
    ax.set_xticks(positions_base)
    ax.set_xticklabels([f'{fid}' for fid in function_ids])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=n_dims, frameon=False)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_by_function_v.pdf', dpi=300, transparent=True)
    plt.savefig(output_dir / 'error_by_function_v.png', dpi=300, transparent=False)
    print(f"Saved: {output_dir / 'error_by_function_v.pdf'}")
    plt.close()

    # 3. Operator usage by dimension (stacked bar with pattern)
    fig, ax = plt.subplots(figsize=(8, 5))

    dims = sorted(df['dimension'].unique())
    operators = ['LF', 'DM', 'PS', 'SP']

    # Prepare data
    usage_data = {op: [] for op in operators}
    for dim in dims:
        df_dim = df[df['dimension'] == dim]
        for op in operators:
            usage_data[op].append(df_dim[f'op_{op}_percent'].mean())

    # Create stacked bar chart
    x = np.arange(len(dims))
    width = 0.6
    bottom = np.zeros(len(dims))

    for op in operators:
        ax.bar(x, usage_data[op], width, label=op, bottom=bottom,
              color=op_colors[op], alpha=0.9, edgecolor='black', linewidth=0.5)
        bottom += np.array(usage_data[op])

    ax.set_ylabel('Operator Usage (\%)')
    ax.set_xlabel('Problem Dimension')
    ax.set_xticks(x)
    ax.set_xticklabels([f'${d}$D' for d in dims])
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', ncol=2, title='Operator')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'operator_usage_by_dimension.pdf', dpi=300, transparent=True)
    plt.savefig(output_dir / 'operator_usage_by_dimension.png', dpi=300, transparent=False)
    print(f"Saved: {output_dir / 'operator_usage_by_dimension.pdf'}")
    plt.close()

    # 4. Error vs operator usage correlation (scatter with regression)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, op in enumerate(['LF', 'DM', 'PS', 'SP']):
        ax = axes[idx]

        # Scatter plot
        ax.scatter(df[f'op_{op}_percent'], df['log_error'],
                  alpha=0.4, s=25, color=op_colors[op], edgecolors='black', linewidth=0.3)

        # Fit regression line
        x_data = df[f'op_{op}_percent'].values
        y_data = df['log_error'].values
        valid = ~(np.isnan(x_data) | np.isnan(y_data))

        if valid.sum() > 2:
            z = np.polyfit(x_data[valid], y_data[valid], 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_line, p(x_line), '--', color='black', linewidth=1.5, alpha=0.7)

        ax.set_xlabel(f'{op} Usage (\%)')
        ax.set_ylabel(r'$\log_{10}(\mathrm{Error})$')

        # Add correlation coefficient
        corr = df[[f'op_{op}_percent', 'log_error']].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f'$r = {corr:.3f}$',
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor='black', alpha=0.9, linewidth=0.8))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_operator_usage.pdf', dpi=300, transparent=True)
    plt.savefig(output_dir / 'error_vs_operator_usage.png', dpi=300, transparent=False)
    print(f"Saved: {output_dir / 'error_vs_operator_usage.pdf'}")
    plt.close()

    # 5. Violin plot of errors by function
    fig, ax = plt.subplots(figsize=(12, 5))

    function_ids = sorted(df['function_id'].unique())
    positions = np.arange(len(function_ids))

    # Prepare data for violin plot
    violin_data = [df[df['function_id'] == fid]['log_error'].values
                   for fid in function_ids]

    parts = ax.violinplot(
        violin_data,
        positions=positions,
        widths=0.7,
        showmeans=False,
        showextrema=False,
        showmedians=True
    )

    # Customize violins
    for pc in parts['bodies']:
        pc.set_facecolor('#009E73')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)

    if 'cmedians' in parts:
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(1.5)

    # Add quartile markers
    for i, fid in enumerate(function_ids):
        data = df[df['function_id'] == fid]['log_error'].values
        q1, q3 = np.percentile(data, [25, 75])
        ax.plot([i, i], [q1, q3], color='black', linewidth=2, alpha=0.8)

    ax.set_xlabel('Function ID')
    ax.set_ylabel(r'$\log_{10}(\mathrm{Error})$')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{fid}' for fid in function_ids])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_violin_by_function.pdf', dpi=300, transparent=True)
    plt.savefig(output_dir / 'error_violin_by_function.png', dpi=300, transparent=False)
    print(f"Saved: {output_dir / 'error_violin_by_function.pdf'}")
    plt.close()

    # 6. Performance comparison heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create pivot table for heatmap
    pivot_data = df.pivot_table(
        values='log_error',
        index='function_id',
        columns='dimension',
        aggfunc='median'
    )

    # Create heatmap
    im = ax.imshow(pivot_data.values, cmap='RdYlGn_r', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels([f'${d}$D' for d in pivot_data.columns])
    ax.set_yticklabels([f'{fid}' for fid in pivot_data.index])

    ax.set_xlabel('Problem Dimension')
    ax.set_ylabel('Function ID')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Median $\log_{10}(\mathrm{Error})$', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.values[i, j]
            text = ax.text(j, i, f'{value:.1f}',
                          ha='center', va='center', color='black', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.pdf', dpi=300, transparent=True)
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, transparent=False)
    print(f"Saved: {output_dir / 'performance_heatmap.pdf'}")
    plt.close()

    print(f"\n✅ All plots saved to: {output_dir}")
    print(f"   - PDF format (vector, transparent background)")
    print(f"   - PNG format (raster, white background)")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main analysis routine"""

    print("="*70)
    print("BATCH RESULTS ANALYSIS")
    print("="*70)
    print(f"\nResults directory: {RESULTS_DIR}")

    if not RESULTS_DIR.exists():
        print(f"\nError: Directory not found: {RESULTS_DIR}")
        sys.exit(1)

    # Load results
    results = load_all_results(RESULTS_DIR)

    if not results:
        print("\nNo results to analyze!")
        sys.exit(1)

    # Create DataFrame
    df = create_dataframe(results)

    # Print summary
    print_summary(df)

    # Create detailed statistics per function-dimension combination
    print("\n" + "="*70)
    print("CREATING FUNCTION-DIMENSION STATISTICS")
    print("="*70)

    stats_df = create_function_dimension_statistics(df)

    if stats_df is not None and len(stats_df) > 0:
        print(f"\nCreated statistics for {len(stats_df)} function-dimension combinations")

        # Display summary of statistics DataFrame
        print("\nStatistics DataFrame columns:")
        print(f"  - Identifiers: function_id, dimension, n_instances")
        print(f"  - Best fitness stats: mean, median, std, min, max, q25, q75")
        print(f"  - Error stats: mean, median, std, min, max, q25, q75")
        print(f"  - Log error stats: mean, median, std")
        print(f"  - Success rates: at 1e-4, 1e-6, 1e-8 thresholds")

        # Save statistics DataFrame
        stats_csv_file = RESULTS_DIR / "function_dimension_statistics.csv"
        stats_df.to_csv(stats_csv_file, index=False)
        print(f"\n✅ Saved function-dimension statistics to: {stats_csv_file}")

        # Print sample of the statistics (first few rows)
        print("\nSample of function-dimension statistics:")
        print(stats_df[['function_id', 'dimension', 'n_instances',
                        'error_mean', 'error_median', 'success_rate_1e-4']].head(10).to_string(index=False))
    else:
        print("⚠️  No statistics to compute")

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PUBLICATION-QUALITY PLOTS")
    print("="*70)
    print("\nPlots will be saved in both PDF (vector) and PNG (raster) formats")
    print("PDF files use transparent backgrounds for easy paper integration\n")
    plot_results(df, RESULTS_DIR / "analysis_plots")

    # Save summary CSV
    csv_file = RESULTS_DIR / "analysis_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSaved summary CSV: {csv_file}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()

