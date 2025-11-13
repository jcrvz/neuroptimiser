#!/bin/bash
#
# Quick launcher for batch experiments
# Usage: ./launch_batch.sh
#

echo "=========================================="
echo "Neuromorphic Optimizer v7 - Batch Runner"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Functions: 1, 2, 8, 10, 15, 17, 20, 21, 24"
echo "  Instances: 1-15"
echo "  Dimensions: 2, 5, 10"
echo "  Total experiments: 270"
echo ""
echo "Estimated runtime: 2-3 hours"
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."

python run_batch_experiments.py

echo ""
echo "=========================================="
echo "Batch experiments complete!"
echo "=========================================="

