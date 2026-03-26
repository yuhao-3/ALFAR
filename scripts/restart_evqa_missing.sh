#!/bin/bash
# Restart any EVQA evaluations that are not complete or running

cd /data/gpfs/projects/punim2075/ALFAR
source ALFAR/bin/activate

echo "Checking and restarting missing EVQA evaluations..."
echo ""

for method in alfar tcvm; do
    for seed in 0 1 2 3 4; do
        preds="experiments/result/multiseed/evqa_${method}_seed${seed}.json"
        log="logs/${method}_evqa_seed${seed}_metrics.txt"

        # Check if already completed
        if grep -q "Accuracy:" "$log" 2>/dev/null; then
            echo "✓ ${method} seed ${seed}: Already complete"
        else
            # Check if currently running
            if ps -ef | grep "python.*${preds}" | grep -v grep > /dev/null; then
                echo "⧗ ${method} seed ${seed}: Already running"
            else
                echo "⧗ ${method} seed ${seed}: Starting..."
                nohup python evaluation/eval_evqa.py --preds "$preds" > "$log" 2>&1 &
                sleep 1
            fi
        fi
    done
done

echo ""
echo "Done!"
echo ""
echo "Running processes:"
ps -ef | grep "python.*eval_evqa" | grep -v grep | wc -l
