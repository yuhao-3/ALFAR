#!/bin/bash
#SBATCH --job-name=reg_mrag_aokvqa
#SBATCH --account=punim2075
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/regular_mrag_aokvqa_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/regular_mrag_aokvqa_%j.err

# CRITICAL MISSING BASELINE - Regular MRAG (Standard RAG)
#
# This provides the control to test ALFAR's amplification effect:
# - Has retrieved context (same as ALFAR)
# - NO contrastive decoding
# - NO attention reallocation
# - Standard generation with context in prompt
#
# Three-way comparison:
# 1. No-Context:    No context + No ALFAR干预
# 2. Regular MRAG:  Has context + No ALFAR干预 (THIS JOB)
# 3. ALFAR:         Has context + Has ALFAR干预
#
# If Regular MRAG > ALFAR in Misleading bucket:
#   → ALFAR amplification causes harm
# If Regular MRAG ≈ ALFAR in Misleading bucket:
#   → Problem is misleading context itself, not ALFAR

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================="

cd /data/gpfs/projects/punim2075/ALFAR
module load CUDA/11.8.0
mkdir -p logs experiments/result
source ALFAR/bin/activate

export CUDA_VISIBLE_DEVICES=0
cd experiments/eval

echo "Starting Regular MRAG (Standard RAG) on A-OKVQA..."

python regular_mrag_llava_okvqa.py \
    --dataset aokvqa \
    --image-folder ../../data/images/coco/val2014 \
    --model-path ../../models/llava-v1.5-7b \
    --answers-file ../result/regular_mrag_aokvqa_results.csv \
    --temperature 1.0 \
    --top_p 1.0 \
    --seed 0

if [ $? -eq 0 ]; then

    echo "Regular MRAG inference completed for A-OKVQA!"

    # Evaluate the results
    cd /data/gpfs/projects/punim2075/ALFAR
    echo "Evaluating Regular MRAG results..."
    python evaluation/eval_okvqa.py \
        --dataset aokvqa \
        --preds experiments/result/regular_mrag_aokvqa_results.csv \
        > logs/regular_mrag_aokvqa_metrics.txt

    echo "All tasks completed!"
    echo "End time: $(date)"
else
    echo "Failed to run Regular MRAG inference"
    exit 1
fi
