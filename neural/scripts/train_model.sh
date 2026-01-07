#!/bin/bash
# Train Neural MMOT Solver
# Usage: bash train_model.sh [--config CONFIG_FILE]

set -e

echo "========================================"
echo "Neural MMOT Solver - Training"
echo "========================================"

# Default configuration
CONFIG_FILE="configs/m4_optimized.yaml"
PYTHON_CMD=python3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration: $CONFIG_FILE"

# Check if data exists
if [ ! -d "data/train" ] || [ -z "$(ls -A data/train 2>/dev/null)" ]; then
    echo ""
    echo "❌ ERROR: Training data not found in data/train/"
    echo ""
    echo "Please generate training data first:"
    echo "  bash scripts/generate_data.sh"
    echo ""
    exit 1
fi

if [ ! -d "data/val" ] || [ -z "$(ls -A data/val 2>/dev/null)" ]; then
    echo ""
    echo "❌ ERROR: Validation data not found in data/val/"
    echo ""
    exit 1
fi

# Count data files
NUM_TRAIN=$(find data/train -name "*.npz" ! -name "._*" | wc -l)
NUM_VAL=$(find data/val -name "*.npz" ! -name "._*" | wc -l)

echo "Training instances: $NUM_TRAIN"
echo "Validation instances: $NUM_VAL"
echo ""

# Create output directories
mkdir -p checkpoints
mkdir -p runs
mkdir -p logs

# Start training
echo "Starting training..."
echo "Started at: $(date)"
echo ""

START_TIME=$(date +%s)

$PYTHON_CMD -u scripts/train.py \
    --config $CONFIG_FILE \
    --train_dir data/train \
    --val_dir data/val \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
RUNTIME_HOURS=$(echo "scale=2; $RUNTIME / 3600" | bc)

echo ""
echo "========================================"
echo "Training complete!"
echo "Finished at: $(date)"
echo "Total runtime: $RUNTIME_HOURS hours"
echo "========================================"
echo ""
echo "Results:"
echo "  Checkpoints: checkpoints/"
echo "  Logs: runs/ (view with tensorboard --logdir runs/)"
echo "  Training log: logs/"
echo ""
echo "Next steps:"
echo "  1. View training curves: tensorboard --logdir runs/"
echo "  2. Evaluate model: bash scripts/evaluate_model.sh"
echo ""
