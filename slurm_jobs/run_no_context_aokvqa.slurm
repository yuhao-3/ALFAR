#!/bin/bash
#SBATCH --job-name=no_ctx_aokvqa
#SBATCH --account=punim2075
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/no_context_aokvqa_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/no_context_aokvqa_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================="

cd /data/gpfs/projects/punim2075/ALFAR
module load CUDA/11.8.0
mkdir -p logs experiments/result
source ALFAR/bin/activate

export CUDA_VISIBLE_DEVICES=0
cd experiments/eval

echo "Starting no-context evaluation on A-OKVQA (Motivation Experiment)..."

python no_context_llava_okvqa.py \
    --dataset aokvqa \
    --image-folder ../../data/images/coco/val2014 \
    --model-path ../../models/llava-v1.5-7b \
    --answers-file ../result/no_context_aokvqa_results.csv \
    --temperature 1.0 \
    --top_p 1.0 \
    --seed 0

if [ $? -eq 0 ]; then
    echo "No-context inference completed for A-OKVQA!"

    # Evaluate the results
    cd /data/gpfs/projects/punim2075/ALFAR
    echo "Evaluating no-context results..."
    python evaluation/eval_okvqa.py \
        --dataset aokvqa \
        --preds experiments/result/no_context_aokvqa_results.csv \
        > logs/no_context_aokvqa_metrics.txt

    echo "All tasks completed!"
    echo "End time: $(date)"
else
    echo "Failed to run no-context inference"
    exit 1
fi
