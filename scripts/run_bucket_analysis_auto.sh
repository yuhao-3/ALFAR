#!/bin/bash
# Automated bucket analysis after no-context job completes
# Run this after job 23306398 finishes

set -e
cd /data/gpfs/projects/punim2075/ALFAR

echo "========================================="
echo "Automated Bucket Analysis Pipeline"
echo "========================================="
echo ""

# Step 1: Verify no-context results exist and have no empty answers
echo "Step 1: Verifying no-context results..."
if [ ! -f "experiments/result/no_context_aokvqa_results.csv" ]; then
    echo "ERROR: No-context results file not found!"
    exit 1
fi

empty_count=$(awk -F, 'NR>1 && $3=="" {count++} END {print count+0}' experiments/result/no_context_aokvqa_results.csv)
total_count=$(awk 'END {print NR-1}' experiments/result/no_context_aokvqa_results.csv)
echo "Empty answers: $empty_count/$total_count"

if [ "$empty_count" -gt 50 ]; then
    echo "WARNING: Too many empty answers! Expected ~0%, got $(awk "BEGIN {print ($empty_count/$total_count)*100}")%"
    echo "Fix may not have worked properly."
fi

echo ""
echo "Sample answers (first 10):"
head -11 experiments/result/no_context_aokvqa_results.csv | tail -10
echo ""

# Step 2: Run bucket analysis
echo "Step 2: Running bucket analysis on A-OKVQA..."
source ALFAR/bin/activate
python scripts/bucket_analysis.py \
    --dataset aokvqa \
    --no-context-file experiments/result/no_context_aokvqa_results.csv \
    --alfar-file experiments/result/aokvqa_alfar_results.csv \
    --question-file data/eval_data/okvqa/a_ok_vqa_val_fixed_annots.csv \
    --output-dir results/bucket_analysis

echo ""
echo "Step 3: Bucket Analysis Results"
echo "================================"
cat results/bucket_analysis/aokvqa_bucket_stats.json | python -m json.tool

echo ""
echo "Step 4: Checking Hypothesis Validation"
echo "======================================="

# Extract delta for Misleading bucket
misleading_delta=$(grep -A10 '"Misleading"' results/bucket_analysis/aokvqa_bucket_stats.json | grep '"delta"' | awk -F: '{print $2}' | tr -d ' ,')

echo "Misleading bucket delta: $misleading_delta%"
if (( $(echo "$misleading_delta < 0" | bc -l) )); then
    echo "✓ HYPOTHESIS VALIDATED! ALFAR underperforms in Misleading bucket"
    echo "  This confirms ALFAR blindly amplifies bad context"
else
    echo "✗ Hypothesis NOT validated. Delta is positive: $misleading_delta%"
    echo "  Need to investigate why ALFAR still helps even with bad context"
fi

echo ""
echo "Step 5: Generating visualization..."
python scripts/plot_motivation.py \
    --bucket-stats results/bucket_analysis/aokvqa_bucket_stats.json \
    --output-dir results/bucket_analysis \
    --dataset aokvqa

echo ""
echo "========================================="
echo "All tasks completed!"
echo "========================================="
echo "Results saved to:"
echo "  - results/bucket_analysis/aokvqa_bucket_stats.json"
echo "  - results/bucket_analysis/aokvqa_bucket_visualization.pdf"
echo "  - results/bucket_analysis/aokvqa_bucket_visualization.png"
