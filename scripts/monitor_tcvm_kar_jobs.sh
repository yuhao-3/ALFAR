#!/bin/bash
# Monitor TCVM-KAR experiment jobs
# Usage: bash scripts/monitor_tcvm_kar_jobs.sh

echo "========================================="
echo "TCVM-KAR Experiment Job Monitor"
echo "========================================="
echo "Timestamp: $(date)"
echo ""

# Job IDs from submission
JOBS=(
    "23132552:infoseek"
    "23132553:viquae"
    "23132554:aokvqa"
    "23132555:okvqa"
    "23132556:evqa"
)

# Check queue status
echo "Current Job Status:"
echo "-------------------"
squeue -u $USER -o "%.10i %.15j %.8T %.10M %.6D %R"
echo ""

# Check individual job statuses
echo "Individual Job Details:"
echo "-----------------------"
for job_info in "${JOBS[@]}"; do
    IFS=':' read -r job_id dataset <<< "$job_info"

    # Get job state
    state=$(squeue -j $job_id -h -o "%T" 2>/dev/null || echo "COMPLETED/FAILED")

    echo "Dataset: $dataset (Job: $job_id)"
    echo "  Status: $state"

    # Check if output file exists and show last few lines
    out_file="logs/tcvm_${dataset}_${job_id}.out"
    if [ -f "$out_file" ]; then
        size=$(du -h "$out_file" | cut -f1)
        lines=$(wc -l < "$out_file")
        echo "  Output: $out_file ($size, $lines lines)"

        # Show last meaningful line
        last_line=$(grep -v "^$" "$out_file" | tail -1 2>/dev/null)
        if [ -n "$last_line" ]; then
            echo "  Last: ${last_line:0:80}..."
        fi
    else
        echo "  Output: Not yet created"
    fi

    # Check if result file exists
    if [ "$dataset" = "infoseek" ] || [ "$dataset" = "viquae" ]; then
        result_file="experiments/result/${dataset}_tcvm_results.jsonl"
    elif [ "$dataset" = "evqa" ]; then
        result_file="experiments/result/${dataset}_tcvm_results.json"
    else
        result_file="experiments/result/${dataset}_tcvm_results.csv"
    fi

    if [ -f "$result_file" ]; then
        result_size=$(du -h "$result_file" | cut -f1)
        echo "  Result: $result_file ($result_size)"
    else
        echo "  Result: Not yet created"
    fi

    echo ""
done

# Summary statistics
echo "Summary:"
echo "--------"
running=$(squeue -u $USER -h -t R | wc -l)
pending=$(squeue -u $USER -h -t PD | wc -l)
total=$((running + pending))

echo "Running: $running"
echo "Pending: $pending"
echo "Total active: $total"
echo ""

if [ $total -eq 0 ]; then
    echo "All jobs completed! Check results in experiments/result/"
    echo ""
    echo "Result files:"
    ls -lh experiments/result/*tcvm_results.* 2>/dev/null || echo "No results found"
    echo ""
    echo "To evaluate results, run:"
    echo "  python evaluation/eval_okvqa.py --dataset aokvqa"
    echo "  python evaluation/eval_okvqa.py --dataset okvqa"
    echo "  python evaluation/eval_mc.py --dataset infoseek"
    echo "  python evaluation/eval_mc.py --dataset viquae"
    echo "  python evaluation/eval_evqa.py"
else
    echo "Jobs still running. Re-run this script to check progress."
    echo ""
    echo "Quick commands:"
    echo "  Watch queue: watch -n 10 'squeue -u \$USER'"
    echo "  Cancel all:  scancel -u \$USER"
fi

echo "========================================="
