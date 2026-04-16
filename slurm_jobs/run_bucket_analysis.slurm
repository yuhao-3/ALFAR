#!/bin/bash
#SBATCH --job-name=bucket_analysis
#SBATCH --account=punim2075
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/bucket_analysis_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/bucket_analysis_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================="

cd /data/gpfs/projects/punim2075/ALFAR
module load CUDA/11.8.0
source ALFAR/bin/activate

mkdir -p results/bucket_analysis figures

echo "Running bucket analysis for motivation experiment..."
echo ""

# InfoSeek bucket analysis
echo "Analyzing InfoSeek..."
python scripts/bucket_analysis.py \
    --dataset infoseek \
    --no-context-file experiments/result/no_context_infoseek_results.jsonl \
    --alfar-file experiments/result/infoseek_alfar_results.jsonl \
    --question-file data/eval_data/mc/infoseek_mc.json \
    --output-dir results/bucket_analysis

if [ $? -eq 0 ]; then
    echo "InfoSeek bucket analysis completed!"

    # Generate plot
    echo "Generating InfoSeek visualization..."
    python scripts/plot_motivation.py \
        --stats-file results/bucket_analysis/infoseek_bucket_stats.json \
        --output-file figures/motivation_infoseek \
        --dataset-name "InfoSeek"
fi

echo ""
echo "----------------------------------------"
echo ""

# A-OKVQA bucket analysis
echo "Analyzing A-OKVQA..."
python scripts/bucket_analysis.py \
    --dataset aokvqa \
    --no-context-file experiments/result/no_context_aokvqa_results.csv \
    --alfar-file experiments/result/aokvqa_alfar_results.csv \
    --question-file data/eval_data/okvqa/a_ok_vqa_val_fixed_annots.csv \
    --output-dir results/bucket_analysis

if [ $? -eq 0 ]; then
    echo "A-OKVQA bucket analysis completed!"

    # Generate plot
    echo "Generating A-OKVQA visualization..."
    python scripts/plot_motivation.py \
        --stats-file results/bucket_analysis/aokvqa_bucket_stats.json \
        --output-file figures/motivation_aokvqa \
        --dataset-name "A-OKVQA"
fi

echo ""
echo "All bucket analyses completed!"
echo "Results saved to: results/bucket_analysis/"
echo "Figures saved to: figures/"
echo "End time: $(date)"
