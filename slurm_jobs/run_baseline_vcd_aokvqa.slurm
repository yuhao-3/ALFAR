#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/baseline_vcd_aokvqa_%j.out
#SBATCH --job-name=vcd_aokvqa

# VCD (Visual Contrastive Decoding) Baseline for A-OKVQA

echo "Starting VCD baseline experiment..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Activate environment
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate

# Run VCD baseline
python experiments/eval/baseline_all_okvqa_llava.py \
    --method vcd \
    --dataset aokvqa \
    --model-path /data/gpfs/projects/punim2075/model/llava_1.5_7b \
    --image-folder /data/gpfs/projects/punim2075/ALFAR/data/images/coco/val2014 \
    --answers-file experiments/result/vcd_aokvqa_results.csv \
    --vcd-alpha 0.5 \
    --vcd-blur-radius 10.0 \
    --temperature 1.0 \
    --top_p 1.0 \
    --seed 0

echo "VCD baseline completed!"
echo "End time: $(date)"

# Evaluate results
echo "Evaluating VCD results..."
python evaluation/eval_okvqa.py \
    --dataset aokvqa \
    --preds experiments/result/vcd_aokvqa_results.csv

echo "Done!"
