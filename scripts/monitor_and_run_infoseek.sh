#!/bin/bash
# Monitor InfoSeek jobs and automatically run analysis when complete

echo "=========================================="
echo "InfoSeek Job Monitor"
echo "=========================================="
echo "Monitoring jobs: 23349669 (No-Context), 23349670 (Regular MRAG)"
echo "Press Ctrl+C to stop monitoring"
echo

while true; do
    # Check job status
    job_status=$(sacct -j 23349669,23349670 --format=State -n | tr -d ' ')

    # Count completed jobs
    completed=$(echo "$job_status" | grep -c "COMPLETED")
    failed=$(echo "$job_status" | grep -c "FAILED")

    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] Status check: $completed completed, $failed failed"

    # Check if both jobs completed
    if [ "$completed" -eq 2 ]; then
        echo
        echo "✓ Both jobs completed successfully!"
        echo "Starting automatic analysis..."
        echo

        cd /data/gpfs/projects/punim2075/ALFAR
        source ALFAR/bin/activate
        bash scripts/run_infoseek_threeway_analysis.sh

        echo
        echo "=========================================="
        echo "Analysis complete! Monitor exiting."
        echo "=========================================="
        exit 0
    fi

    # Check if any job failed
    if [ "$failed" -gt 0 ]; then
        echo
        echo "❌ One or more jobs failed!"
        echo "Check logs:"
        echo "  tail logs/no_context_infoseek_23349669.err"
        echo "  tail logs/regular_mrag_infoseek_23349670.err"
        exit 1
    fi

    # Wait 5 minutes before next check
    sleep 300
done
