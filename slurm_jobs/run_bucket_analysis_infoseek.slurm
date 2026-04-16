#!/bin/bash
#SBATCH --job-name=bucket_infoseek
#SBATCH --account=punim2075
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/bucket_infoseek_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/bucket_infoseek_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================="

cd /data/gpfs/projects/punim2075/ALFAR
source ALFAR/bin/activate

echo "Running bucket analysis for InfoSeek..."
python scripts/bucket_analysis_infoseek.py \
    --no-context-file experiments/result/no_context_infoseek_results.jsonl \
    --regular-mrag-file experiments/result/regular_mrag_infoseek_results.jsonl \
    --alfar-file experiments/result/infoseek_alfar_results.jsonl \
    --question-file data/eval_data/mc/infoseek_mc.json \
    --output-dir results/bucket_analysis_threeway

if [ $? -eq 0 ]; then
    echo "Bucket analysis completed successfully!"

    echo "Generating visualizations..."
    python scripts/plot_threeway_bucket_analysis_infoseek.py

    if [ $? -eq 0 ]; then
        echo "Visualizations generated!"
        echo "Results:"
        ls -lh results/bucket_analysis_threeway/infoseek_*

        echo ""
        echo "Summary statistics:"
        cat results/bucket_analysis_threeway/infoseek_bucket_stats.json
    else
        echo "Visualization generation failed"
        exit 1
    fi
else
    echo "Bucket analysis failed!"
    exit 1
fi

echo "End time: $(date)"
