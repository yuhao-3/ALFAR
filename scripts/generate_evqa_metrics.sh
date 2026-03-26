#!/bin/bash
# Generate all EVQA metrics
cd /data/gpfs/projects/punim2075/ALFAR
source ALFAR/bin/activate

echo "Generating EVQA metrics (this may take time for TensorFlow Hub model download)..."

for seed in 0 1 2 3 4; do
    echo "Processing ALFAR E-VQA seed $seed..."
    python evaluation/eval_evqa.py \
        --preds experiments/result/multiseed/evqa_alfar_seed${seed}.json \
        > logs/alfar_evqa_seed${seed}_metrics.txt 2>&1 &
done

for seed in 0 1 2 3 4; do
    echo "Processing TCVM E-VQA seed $seed..."
    python evaluation/eval_evqa.py \
        --preds experiments/result/multiseed/evqa_tcvm_seed${seed}.json \
        > logs/tcvm_evqa_seed${seed}_metrics.txt 2>&1 &
done

echo "All EVQA evaluations started in background. Waiting for completion..."
wait

echo "EVQA metrics generation complete!"
echo "Checking results..."
grep -h "Accuracy" logs/*evqa_seed*_metrics.txt 2>/dev/null | sort
