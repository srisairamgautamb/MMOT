#!/bin/bash
# Generate MMOT training dataset using Phase 2a solver
# Runtime: ~50 hours on Apple M4

set -e  # Exit on error

echo "========================================"
echo "MMOT Dataset Generation"
echo "========================================"

# Configuration
NUM_INSTANCES=10000
START_IDX=0
PYTHON_CMD=python3

# Check if Phase 2a solver is available
echo "Checking Phase 2a solver..."
if [ ! -d "../../mmot_jax" ]; then
    echo "WARNING: Phase 2a solver not found at ../../mmot_jax"
    echo "Please ensure Phase 2a solver is available before running."
    exit 1
fi

# Create output directories
echo "Creating output directories..."
mkdir -p ../data/raw
mkdir -p ../data/train
mkdir -p ../data/val

# Estimate runtime
ESTIMATED_HOURS=$(echo "scale=1; $NUM_INSTANCES * 5 / 3600" | bc)
echo ""
echo "Configuration:"
echo "  Instances: $NUM_INSTANCES"
echo "  Estimated time: $ESTIMATED_HOURS hours"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Start generation
echo ""
echo "Starting dataset generation..."
echo "Started at: $(date)"
START_TIME=$(date +%s)

$PYTHON_CMD ../data/generator.py --num $NUM_INSTANCES --start $START_IDX

# Calculate actual runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
RUNTIME_HOURS=$(echo "scale=2; $RUNTIME / 3600" | bc)

echo ""
echo "========================================"
echo "Dataset generation complete!"
echo "Finished at: $(date)"
echo "Total runtime: $RUNTIME_HOURS hours"
echo "========================================"
