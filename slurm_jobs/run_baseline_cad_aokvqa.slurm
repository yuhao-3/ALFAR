#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/baseline_cad_aokvqa_%j.out
#SBATCH --job-name=cad_aokvqa

# CAD (Context-Aware Decoding) Baseline for A-OKVQA

echo "Starting CAD baseline experiment..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Activate environment
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate

# Run CAD baseline
python experiments/eval/baseline_all_okvqa_llava.py \
    --method cad \
    --dataset aokvqa \
    --model-path /data/gpfs/projects/punim2075/model/llava_1.5_7b \
    --image-folder /data/gpfs/projects/punim2075/ALFAR/data/images/coco/val2014 \
    --answers-file experiments/result/cad_aokvqa_results.csv \
    --cad-alpha 0.5 \
    --temperature 1.0 \
    --top_p 1.0 \
    --seed 0

echo "CAD baseline completed!"
echo "End time: $(date)"

# Evaluate results
echo "Evaluating CAD results..."
python evaluation/eval_okvqa.py \
    --dataset aokvqa \
    --preds experiments/result/cad_aokvqa_results.csv

echo "Done!"
