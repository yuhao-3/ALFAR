#!/bin/bash
#SBATCH --job-name=no_ctx_infoseek
#SBATCH --account=punim2075
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/no_context_infoseek_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/no_context_infoseek_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================="

cd /data/gpfs/projects/punim2075/ALFAR
module load CUDA/11.8.0
mkdir -p logs experiments/result
source ALFAR/bin/activate

export CUDA_VISIBLE_DEVICES=0
cd experiments/eval

echo "Starting No-Context baseline on InfoSeek..."

python no_context_llava.py \
    --dataset infoseek \
    --image-folder ../../data/images/infoseek_images \
    --model-path ../../models/llava-v1.5-7b \
    --answers-file ../result/no_context_infoseek_results.jsonl \
    --temperature 1.0 \
    --top_p 1.0 \
    --seed 0

if [ $? -eq 0 ]; then
    echo "No-Context inference completed for InfoSeek!"

    # Evaluate the results
    cd /data/gpfs/projects/punim2075/ALFAR
    echo "Evaluating No-Context results..."
    python evaluation/eval_mc.py \
        --dataset infoseek \
        --preds experiments/result/no_context_infoseek_results.jsonl \
        > logs/no_context_infoseek_metrics.txt

    echo "All tasks completed!"
    echo "End time: $(date)"
else
    echo "Failed to run No-Context inference"
    exit 1
fi
