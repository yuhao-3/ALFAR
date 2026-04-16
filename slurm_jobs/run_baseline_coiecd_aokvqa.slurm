#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/baseline_coiecd_aokvqa_%j.out
#SBATCH --job-name=coiecd_aokvqa

# COIECD (Contextual Information-Entropy Constraint Decoding) Baseline for A-OKVQA

echo "Starting COIECD baseline experiment..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Activate environment
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate

# Run COIECD baseline
python experiments/eval/baseline_all_okvqa_llava.py \
    --method coiecd \
    --dataset aokvqa \
    --model-path /data/gpfs/projects/punim2075/model/llava_1.5_7b \
    --image-folder /data/gpfs/projects/punim2075/ALFAR/data/images/coco/val2014 \
    --answers-file experiments/result/coiecd_aokvqa_results.csv \
    --coiecd-alpha 0.5 \
    --coiecd-temperature 0.7 \
    --top_p 1.0 \
    --seed 0

echo "COIECD baseline completed!"
echo "End time: $(date)"

# Evaluate results
echo "Evaluating COIECD results..."
python evaluation/eval_okvqa.py \
    --dataset aokvqa \
    --preds experiments/result/coiecd_aokvqa_results.csv

echo "Done!"
