#!/bin/bash

# Navigate to the workspace root explicitly utilizing the script's relative path 
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKSPACE_DIR"

# Ensure log directory exists
LOG_DIR="logs/benchmarks"
mkdir -p "$LOG_DIR"

# Generate a unique log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/benchmark_${TIMESTAMP}.log"

echo "---------------------------------------------------------"
echo "🚀 Starting HeteroGNN Benchmark Suite"
echo "📂 Saving Output Log to: $LOG_FILE"
echo "---------------------------------------------------------"

# Run with unbuffered python standard output mapped seamlessly through `tee`
# Passing all bash script arguments directly to the python script using "$@"
PYTHONUNBUFFERED=1 python3 lipid_gnn/benchmark_heterognn.py "$@" 2>&1 | tee "$LOG_FILE"

echo "---------------------------------------------------------"
echo "✅ Benchmark execution complete! Output stored persistently at: $LOG_FILE"
