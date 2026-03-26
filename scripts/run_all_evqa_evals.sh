#!/bin/bash
# Run all EVQA evaluations with proper environment

cd /data/gpfs/projects/punim2075/ALFAR
source ALFAR/bin/activate

echo "================================"
echo "Starting All E-VQA Evaluations"
echo "================================"
echo "Estimated time: ~6-7 hours total"
echo ""

# Start all evaluations in background
for method in alfar tcvm; do
    for seed in 0 1 2 3 4; do
        log_file="logs/${method}_evqa_seed${seed}_metrics.txt"
        pred_file="experiments/result/multiseed/evqa_${method}_seed${seed}.json"

        echo "Starting: ${method} seed ${seed}"
        nohup python evaluation/eval_evqa.py \
            --preds "$pred_file" \
            > "$log_file" 2>&1 &

        # Give it a moment to start
        sleep 2
    done
done

echo ""
echo "All 10 evaluations started!"
echo ""
echo "To monitor progress:"
echo "  bash scripts/check_evqa_progress.sh"
echo ""
echo "Running processes:"
ps aux | grep eval_evqa.py | grep -v grep | wc -l
