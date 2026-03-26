#!/bin/bash
#SBATCH --job-name=tcvm_evqa_ms
#SBATCH --output=logs/tcvm_evqa_seed%a_%j.out
#SBATCH --error=logs/tcvm_evqa_seed%a_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --array=1-2

echo "========================================="
echo "Job ID: $SLURM_JOB_ID | Seed: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date)"
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
cd /data/gpfs/projects/punim2075/ALFAR
mkdir -p results/llava1.5/evqa

python experiments/eval/alfar_evqa_llava.py \
    --use_tcvm --tcvm_topk 20 --tcvm_alpha 1.0 --tcvm_beta 0.7 \
    --seed $SLURM_ARRAY_TASK_ID \
    --answers-file results/llava1.5/evqa/evqa_tcvm_seed${SLURM_ARRAY_TASK_ID}_results.json \
    --image-folder data/images/oven_eval/ \
    --model-path models/llava-v1.5-7b

echo "End: $(date)"
