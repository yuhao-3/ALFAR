#!/bin/bash
# Extract accuracies from all completed multiseed experiments

cd /data/gpfs/projects/punim2075/ALFAR
source ALFAR/bin/activate

echo "Extracting accuracies from completed experiments..."
echo "=================================================="
echo ""

# A-OKVQA ALFAR
echo "A-OKVQA ALFAR:"
for seed in 0 1 2 3 4; do
    acc=$(python evaluation/eval_okvqa.py --dataset aokvqa --preds experiments/result/multiseed/aokvqa_alfar_seed${seed}.csv 2>&1 | grep "AOKVQA Accuracy:" | awk '{print $3}')
    echo "  Seed ${seed}: ${acc}"
done
echo ""

# A-OKVQA TCVM
echo "A-OKVQA TCVM:"
for seed in 0 1 2 3 4; do
    acc=$(python evaluation/eval_okvqa.py --dataset aokvqa --preds experiments/result/multiseed/aokvqa_tcvm_seed${seed}.csv 2>&1 | grep "AOKVQA Accuracy:" | awk '{print $3}')
    echo "  Seed ${seed}: ${acc}"
done
echo ""

# OKVQA ALFAR
echo "OKVQA ALFAR:"
for seed in 0 1 2 3 4; do
    acc=$(python evaluation/eval_okvqa.py --dataset okvqa --preds experiments/result/multiseed/okvqa_alfar_seed${seed}.csv 2>&1 | grep "OKVQA Accuracy:" | awk '{print $3}')
    echo "  Seed ${seed}: ${acc}"
done
echo ""

# OKVQA TCVM
echo "OKVQA TCVM:"
for seed in 0 1 2 3 4; do
    acc=$(python evaluation/eval_okvqa.py --dataset okvqa --preds experiments/result/multiseed/okvqa_tcvm_seed${seed}.csv 2>&1 | grep "OKVQA Accuracy:" | awk '{print $3}')
    echo "  Seed ${seed}: ${acc}"
done
echo ""

# InfoSeek ALFAR
echo "InfoSeek ALFAR:"
for seed in 0 1 2 3 4; do
    if [ -s experiments/result/multiseed/infoseek_alfar_seed${seed}.jsonl ]; then
        acc=$(python evaluation/eval_mc.py --dataset infoseek --preds experiments/result/multiseed/infoseek_alfar_seed${seed}.jsonl 2>&1 | grep "Accuracy:" | awk '{print $2}')
        echo "  Seed ${seed}: ${acc}"
    fi
done
echo ""

# InfoSeek TCVM
echo "InfoSeek TCVM:"
for seed in 0 1 2 3 4; do
    if [ -s experiments/result/multiseed/infoseek_tcvm_seed${seed}.jsonl ]; then
        acc=$(python evaluation/eval_mc.py --dataset infoseek --preds experiments/result/multiseed/infoseek_tcvm_seed${seed}.jsonl 2>&1 | grep "Accuracy:" | awk '{print $2}')
        echo "  Seed ${seed}: ${acc}"
    fi
done
echo ""

# E-VQA ALFAR
echo "E-VQA ALFAR:"
for seed in 0 1 2 3 4; do
    if [ -s experiments/result/multiseed/evqa_alfar_seed${seed}.json ]; then
        acc=$(python evaluation/eval_evqa.py --preds experiments/result/multiseed/evqa_alfar_seed${seed}.json 2>&1 | tail -1 | awk '{print $NF}')
        echo "  Seed ${seed}: ${acc}"
    fi
done
echo ""

# E-VQA TCVM
echo "E-VQA TCVM:"
for seed in 0 1 2 3 4; do
    if [ -s experiments/result/multiseed/evqa_tcvm_seed${seed}.json ]; then
        acc=$(python evaluation/eval_evqa.py --preds experiments/result/multiseed/evqa_tcvm_seed${seed}.json 2>&1 | tail -1 | awk '{print $NF}')
        echo "  Seed ${seed}: ${acc}"
    fi
done
echo ""

# ViQuAE ALFAR
echo "ViQuAE ALFAR:"
for seed in 0 1 2 3 4; do
    if [ -s experiments/result/multiseed/viquae_alfar_seed${seed}.jsonl ]; then
        acc=$(python evaluation/eval_mc.py --dataset viquae --preds experiments/result/multiseed/viquae_alfar_seed${seed}.jsonl 2>&1 | grep "Accuracy:" | awk '{print $2}')
        echo "  Seed ${seed}: ${acc}"
    fi
done
echo ""

# ViQuAE TCVM
echo "ViQuAE TCVM:"
for seed in 0 1 2 3 4; do
    if [ -s experiments/result/multiseed/viquae_tcvm_seed${seed}.jsonl ]; then
        acc=$(python evaluation/eval_mc.py --dataset viquae --preds experiments/result/multiseed/viquae_tcvm_seed${seed}.jsonl 2>&1 | grep "Accuracy:" | awk '{print $2}')
        echo "  Seed ${seed}: ${acc}"
    fi
done
