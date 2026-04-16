#!/bin/bash
# Full pipeline for InfoSeek three-way bucket analysis
# Run this after jobs 23349669 and 23349670 complete

set -e  # Exit on error

echo "=========================================="
echo "InfoSeek Three-Way Bucket Analysis Pipeline"
echo "=========================================="
echo

# Check if result files exist
echo "Step 1: Checking if inference results exist..."
if [ ! -s experiments/result/no_context_infoseek_results.jsonl ]; then
    echo "ERROR: No-Context results file is empty or missing!"
    echo "Wait for job 23349669 to complete."
    exit 1
fi

if [ ! -s experiments/result/regular_mrag_infoseek_results.jsonl ]; then
    echo "ERROR: Regular MRAG results file is empty or missing!"
    echo "Wait for job 23349670 to complete."
    exit 1
fi

if [ ! -s experiments/result/infoseek_alfar_results.jsonl ]; then
    echo "ERROR: ALFAR results file is empty or missing!"
    exit 1
fi

echo "✓ All result files found"
echo

# Count lines
echo "Result file sizes:"
wc -l experiments/result/no_context_infoseek_results.jsonl
wc -l experiments/result/regular_mrag_infoseek_results.jsonl
wc -l experiments/result/infoseek_alfar_results.jsonl
echo

# Run bucket analysis
echo "Step 2: Running bucket analysis..."
python scripts/bucket_analysis_infoseek.py \
    --no-context-file experiments/result/no_context_infoseek_results.jsonl \
    --regular-mrag-file experiments/result/regular_mrag_infoseek_results.jsonl \
    --alfar-file experiments/result/infoseek_alfar_results.jsonl \
    --question-file data/eval_data/mc/infoseek_mc.json \
    --output-dir results/bucket_analysis_threeway

if [ $? -ne 0 ]; then
    echo "ERROR: Bucket analysis failed!"
    exit 1
fi
echo

# Generate visualizations
echo "Step 3: Generating visualizations..."
python scripts/plot_threeway_bucket_analysis_infoseek.py

if [ $? -ne 0 ]; then
    echo "ERROR: Visualization generation failed!"
    exit 1
fi
echo

echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo
echo "Results saved to:"
echo "  - results/bucket_analysis_threeway/infoseek_bucket_stats.json"
echo "  - results/bucket_analysis_threeway/infoseek_bucket_assignments.json"
echo "  - results/bucket_analysis_threeway/infoseek_bucket_samples.json"
echo "  - results/bucket_analysis_threeway/infoseek_threeway_comparison.png"
echo "  - results/bucket_analysis_threeway/infoseek_delta_comparison.png"
echo

# Display key findings
echo "Key Findings:"
echo "─────────────────────────────────────────"
cat results/bucket_analysis_threeway/infoseek_bucket_stats.json | python -m json.tool | grep -A 10 "Misleading"
echo
