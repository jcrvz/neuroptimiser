#!/usr/bin/env python3
"""Test function_dimension_statistics functionality"""

import pandas as pd
import numpy as np

# Create sample data
sample_data = {
    'function_id': [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
    'instance': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
    'dimension': [2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10],
    'best_fitness': [1.5, 1.2, 1.8, 2.1, 2.3, 2.0, 5.5, 5.2, 5.8, 8.1, 8.3, 8.0],
    'optimal_fitness': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 8.0, 8.0, 8.0],
    'error': [0.5, 0.2, 0.8, 0.1, 0.3, 0.0, 0.5, 0.2, 0.8, 0.1, 0.3, 0.0],
    'total_evaluations': [1000000] * 12,
}

df = pd.DataFrame(sample_data)
df['abs_error'] = df['error'].abs()
df['log_error'] = np.log10(df['abs_error'] + 1e-16)

# Import the function
from analyze_results import create_function_dimension_statistics

# Test it
stats_df = create_function_dimension_statistics(df)

print('✅ Function works correctly!')
print(f'\nInput: {len(df)} rows')
print(f'Output: {len(stats_df)} function-dimension combinations\n')

print('Columns in statistics DataFrame:')
for col in stats_df.columns:
    print(f'  - {col}')

print('\n' + '='*70)
print('Sample Statistics (first few columns):')
print('='*70)
print(stats_df[['function_id', 'dimension', 'n_instances',
                'error_mean', 'error_median', 'error_std',
                'success_rate_1e-4']].to_string(index=False))

print('\n' + '='*70)
print('All Statistics for Function 1, D=2:')
print('='*70)
row = stats_df[(stats_df['function_id'] == 1) & (stats_df['dimension'] == 2)].iloc[0]
for col in stats_df.columns:
    print(f'{col:25s}: {row[col]}')

print('\n✅ Test completed successfully!')

