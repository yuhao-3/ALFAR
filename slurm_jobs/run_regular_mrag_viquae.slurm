#!/bin/bash
#SBATCH --job-name=reg_mrag_viquae
#SBATCH --account=punim2075
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/regular_mrag_viquae_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/regular_mrag_viquae_%j.err

# Regular MRAG (Standard RAG) for ViQuAE
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

echo "Starting Regular MRAG (Standard RAG) on ViQuAE..."

python regular_mrag_llava_mc.py \
    --dataset viquae \
    --image-folder ../../data/images/viquae \
    --model-path ../../models/llava-v1.5-7b \
    --answers-file ../result/regular_mrag_viquae_results.jsonl \
    --temperature 1.0 \
    --top_p 1.0 \
    --seed 0

if [ $? -eq 0 ]; then

    echo "Regular MRAG inference completed for ViQuAE!"

    # Evaluate the results
    cd /data/gpfs/projects/punim2075/ALFAR
    echo "Evaluating Regular MRAG results..."
    python evaluation/eval_mc.py \
        --dataset viquae \
        --preds experiments/result/regular_mrag_viquae_results.jsonl \
        > logs/regular_mrag_viquae_metrics.txt

    echo "All tasks completed!"
    echo "End time: $(date)"
else
    echo "Failed to run Regular MRAG inference"
    exit 1
fi
