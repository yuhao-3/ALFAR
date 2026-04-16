#!/bin/bash
# Quick Reference: Run All Baselines for Comparison
# This script runs all baseline methods on A-OKVQA for easy comparison

set -e  # Exit on error

DATASET="aokvqa"
MODEL_PATH="/data/gpfs/projects/punim2075/model/llava_1.5_7b"
IMAGE_FOLDER="/data/gpfs/projects/punim2075/ALFAR/data/images/coco/val2014"
SEED=0

echo "======================================"
echo "ALFAR Baseline Comparison Experiment"
echo "======================================"
echo "Dataset: $DATASET"
echo "Model: LLaVA-1.5-7B"
echo "Seed: $SEED"
echo ""

# Activate environment
echo "Activating environment..."
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate

cd /data/gpfs/projects/punim2075/ALFAR

# Create results directory
mkdir -p experiments/result
mkdir -p logs/baselines

echo ""
echo "======================================"
echo "Running Baseline Methods"
echo "======================================"

# 1. VCD - Visual Contrastive Decoding
echo ""
echo "[1/6] Running VCD (Visual Contrastive Decoding)..."
python experiments/eval/baseline_all_okvqa_llava.py \
    --method vcd \
    --dataset $DATASET \
    --model-path $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --answers-file experiments/result/vcd_${DATASET}_results.csv \
    --vcd-alpha 0.5 \
    --vcd-blur-radius 10.0 \
    --seed $SEED \
    2>&1 | tee logs/baselines/vcd_${DATASET}.log

# 2. CD - Contrastive Decoding
echo ""
echo "[2/6] Running CD (Contrastive Decoding)..."
python experiments/eval/baseline_all_okvqa_llava.py \
    --method cd \
    --dataset $DATASET \
    --model-path $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --answers-file experiments/result/cd_${DATASET}_results.csv \
    --cd-alpha 0.5 \
    --seed $SEED \
    2>&1 | tee logs/baselines/cd_${DATASET}.log

# 3. CAD - Context-Aware Decoding
echo ""
echo "[3/6] Running CAD (Context-Aware Decoding)..."
python experiments/eval/baseline_all_okvqa_llava.py \
    --method cad \
    --dataset $DATASET \
    --model-path $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --answers-file experiments/result/cad_${DATASET}_results.csv \
    --cad-alpha 0.5 \
    --seed $SEED \
    2>&1 | tee logs/baselines/cad_${DATASET}.log

# 4. AdaCAD - Adaptive Context-Aware Decoding
echo ""
echo "[4/6] Running AdaCAD (Adaptive Context-Aware Decoding)..."
python experiments/eval/baseline_all_okvqa_llava.py \
    --method adacad \
    --dataset $DATASET \
    --model-path $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --answers-file experiments/result/adacad_${DATASET}_results.csv \
    --adacad-alpha-max 1.0 \
    --seed $SEED \
    2>&1 | tee logs/baselines/adacad_${DATASET}.log

# 5. Entropy - Entropy-based Decoding
echo ""
echo "[5/6] Running Entropy-based Decoding..."
python experiments/eval/baseline_all_okvqa_llava.py \
    --method entropy \
    --dataset $DATASET \
    --model-path $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --answers-file experiments/result/entropy_${DATASET}_results.csv \
    --entropy-temperature 0.5 \
    --seed $SEED \
    2>&1 | tee logs/baselines/entropy_${DATASET}.log

# 6. COIECD - Contextual Information-Entropy Constraint Decoding
echo ""
echo "[6/6] Running COIECD..."
python experiments/eval/baseline_all_okvqa_llava.py \
    --method coiecd \
    --dataset $DATASET \
    --model-path $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --answers-file experiments/result/coiecd_${DATASET}_results.csv \
    --coiecd-alpha 0.5 \
    --coiecd-temperature 0.7 \
    --seed $SEED \
    2>&1 | tee logs/baselines/coiecd_${DATASET}.log

echo ""
echo "======================================"
echo "Evaluating All Baselines"
echo "======================================"

# Evaluate each baseline
for method in vcd cd cad adacad entropy coiecd; do
    echo ""
    echo "Evaluating $method..."
    python evaluation/eval_okvqa.py \
        --dataset $DATASET \
        --preds experiments/result/${method}_${DATASET}_results.csv \
        2>&1 | tee logs/baselines/${method}_${DATASET}_eval.log
done

echo ""
echo "======================================"
echo "All Baselines Completed!"
echo "======================================"
echo ""
echo "Results saved to: experiments/result/"
echo "Logs saved to: logs/baselines/"
echo ""
echo "Next steps:"
echo "1. Compare results with: experiments/result/*_${DATASET}_results.csv"
echo "2. Run bucket analysis to understand performance by context quality"
echo "3. Compare with ALFAR and No-Context baselines"
echo ""
echo "Done!"
