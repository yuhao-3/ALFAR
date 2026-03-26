#!/bin/bash
#SBATCH --job-name=alfar_okvqa_multiseed
#SBATCH --account=punim2075
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/alfar_okvqa_seed%a_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/alfar_okvqa_seed%a_%j.err

SEED=$SLURM_ARRAY_TASK_ID

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Start time: $(date)"
echo "========================================="

cd /data/gpfs/projects/punim2075/ALFAR
module load CUDA/11.8.0
mkdir -p logs experiments/result/multiseed
source ALFAR/bin/activate

export CUDA_VISIBLE_DEVICES=0
cd experiments/eval

echo "Starting ALFAR evaluation on OK-VQA with seed=${SEED}..."
echo "ALFAR Configuration:"
echo "  - Attention alpha: 0.4"
echo "  - CD beta: 0.7"
echo "  - Retrieval similarity: 0.9"

python alfar_okvqa_llava.py \
    --dataset okvqa \
    --image-folder ../../data/images/coco/val2014 \
    --model-path ../../models/llava-v1.5-7b \
    --answers-file ../result/multiseed/okvqa_alfar_seed${SEED}.csv \
    --temperature 1.0 \
    --top_p 1.0 \
    --cd_beta 0.7 \
    --att_alpha 0.4 \
    --seed ${SEED}

if [ $? -eq 0 ]; then
    echo "Prediction completed for seed ${SEED}!"
    cd /data/gpfs/projects/punim2075/ALFAR
    python evaluation/eval_okvqa.py \
        --dataset okvqa \
        --preds experiments/result/multiseed/okvqa_alfar_seed${SEED}.csv \
        > logs/alfar_okvqa_seed${SEED}_metrics.txt
    echo "All tasks completed for seed ${SEED}!"
    echo "End time: $(date)"
else
    echo "Failed for seed ${SEED}"
    exit 1
fi
