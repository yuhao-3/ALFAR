#!/bin/bash
#SBATCH --job-name=reg_mrag_okvqa
#SBATCH --account=punim2075
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/regular_mrag_okvqa_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/regular_mrag_okvqa_%j.err

# Regular MRAG (Standard RAG) for OK-VQA
#
# Three-way comparison:
# 1. No-Context:    No context + No ALFAR
# 2. Regular MRAG:  Has context + No ALFAR (THIS JOB)
# 3. ALFAR:         Has context + Has ALFAR

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================="

cd /data/gpfs/projects/punim2075/ALFAR
module load CUDA/11.8.0
mkdir -p logs experiments/result
source ALFAR/bin/activate

export CUDA_VISIBLE_DEVICES=0
cd experiments/eval

echo "Starting Regular MRAG (Standard RAG) on OK-VQA..."

python regular_mrag_llava_okvqa.py \
    --dataset okvqa \
    --image-folder ../../data/images/coco/val2014 \
    --model-path ../../models/llava-v1.5-7b \
    --answers-file ../result/regular_mrag_okvqa_results.csv \
    --temperature 1.0 \
    --top_p 1.0 \
    --seed 0

if [ $? -eq 0 ]; then

    echo "Regular MRAG inference completed for OK-VQA!"

    # Evaluate the results
    cd /data/gpfs/projects/punim2075/ALFAR
    echo "Evaluating Regular MRAG results..."
    python evaluation/eval_okvqa.py \
        --dataset okvqa \
        --preds experiments/result/regular_mrag_okvqa_results.csv \
        > logs/regular_mrag_okvqa_metrics.txt

    echo "All tasks completed!"
    echo "End time: $(date)"
else
    echo "Failed to run Regular MRAG inference"
    exit 1
fi
