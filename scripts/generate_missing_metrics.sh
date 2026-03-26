#!/bin/bash
# Generate missing metrics files from prediction results

cd /data/gpfs/projects/punim2075/ALFAR
source ALFAR/bin/activate

echo "Generating missing metrics..."
echo "================================"

# A-OKVQA metrics
for seed in 0 1 2 3 4; do
    echo "Processing ALFAR A-OKVQA seed $seed..."
    if [ -f "experiments/result/multiseed/aokvqa_alfar_seed${seed}.csv" ]; then
        python evaluation/eval_okvqa.py \
            --dataset aokvqa \
            --preds experiments/result/multiseed/aokvqa_alfar_seed${seed}.csv \
            > logs/alfar_aokvqa_seed${seed}_metrics.txt 2>&1
    fi

    echo "Processing TCVM A-OKVQA seed $seed..."
    if [ -f "experiments/result/multiseed/aokvqa_tcvm_seed${seed}.csv" ]; then
        python evaluation/eval_okvqa.py \
            --dataset aokvqa \
            --preds experiments/result/multiseed/aokvqa_tcvm_seed${seed}.csv \
            > logs/tcvm_aokvqa_seed${seed}_metrics.txt 2>&1
    fi
done

# OKVQA metrics
for seed in 0 1 2 3 4; do
    echo "Processing ALFAR OKVQA seed $seed..."
    if [ -f "experiments/result/multiseed/okvqa_alfar_seed${seed}.csv" ]; then
        python evaluation/eval_okvqa.py \
            --dataset okvqa \
            --preds experiments/result/multiseed/okvqa_alfar_seed${seed}.csv \
            > logs/alfar_okvqa_seed${seed}_metrics.txt 2>&1
    fi

    echo "Processing TCVM OKVQA seed $seed..."
    if [ -f "experiments/result/multiseed/okvqa_tcvm_seed${seed}.csv" ]; then
        python evaluation/eval_okvqa.py \
            --dataset okvqa \
            --preds experiments/result/multiseed/okvqa_tcvm_seed${seed}.csv \
            > logs/tcvm_okvqa_seed${seed}_metrics.txt 2>&1
    fi
done

# EVQA metrics
for seed in 0 1 2 3 4; do
    echo "Processing ALFAR EVQA seed $seed..."
    if [ -f "experiments/result/multiseed/evqa_alfar_seed${seed}.json" ]; then
        python evaluation/eval_evqa.py \
            --dataset evqa \
            --preds experiments/result/multiseed/evqa_alfar_seed${seed}.json \
            > logs/alfar_evqa_seed${seed}_metrics.txt 2>&1
    fi

    echo "Processing TCVM EVQA seed $seed..."
    if [ -f "experiments/result/multiseed/evqa_tcvm_seed${seed}.json" ]; then
        python evaluation/eval_evqa.py \
            --dataset evqa \
            --preds experiments/result/multiseed/evqa_tcvm_seed${seed}.json \
            > logs/tcvm_evqa_seed${seed}_metrics.txt 2>&1
    fi
done

echo "================================"
echo "Metrics generation complete!"
echo ""
echo "Checking which result files exist:"
ls -1 experiments/result/multiseed/ | grep -E "(aokvqa|okvqa|evqa)" | sort
