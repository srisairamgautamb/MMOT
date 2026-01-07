#!/bin/bash
# Launch 8 parallel workers to generate 10,000 samples
# Total time: ~4-5 hours (parallel) vs 30 hours (serial)

mkdir -p neural/logs
mkdir -p neural/data/raw

echo "=================================================="
echo "LAUNCHING PARALLEL DATA GENERATION (10k Samples)"
echo "=================================================="

N_WORKERS=8
SAMPLES_PER_WORKER=1250

for i in $(seq 0 $((N_WORKERS-1))); do
    START_IDX=$((i * SAMPLES_PER_WORKER))
    LOG_FILE="neural/logs/worker_${i}.log"
    
    echo "Worker $i: Generating $SAMPLES_PER_WORKER samples (Start: $START_IDX) -> $LOG_FILE"
    
    # Run in background
    python3 neural/data/generator.py --num $SAMPLES_PER_WORKER --start $START_IDX > $LOG_FILE 2>&1 &
done

echo "=================================================="
echo "All 8 workers launched in background."
echo "Monitor progress via: tail -f neural/logs/worker_*.log"
echo "=================================================="
