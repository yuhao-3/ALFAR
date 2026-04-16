#!/bin/bash
# Watch job 23306398 and automatically run analysis when complete

JOB_ID=23306398
LOG_FILE="logs/no_context_aokvqa_${JOB_ID}.out"
CHECK_INTERVAL=30  # seconds

echo "Watching job $JOB_ID..."
echo "Will automatically run bucket analysis when job completes"
echo "Press Ctrl+C to stop watching"
echo ""

while true; do
    # Check if job is still in queue
    if squeue -j $JOB_ID 2>/dev/null | grep -q $JOB_ID; then
        status=$(squeue -j $JOB_ID --format="%T" | tail -1)
        echo "[$(date +%H:%M:%S)] Job status: $status"
    else
        # Job not in queue - check if it completed
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "[$(date +%H:%M:%S)] Job completed! Log file found."
            echo "Waiting 5 seconds for file writes to finish..."
            sleep 5

            echo ""
            echo "========================================="
            echo "Running automated bucket analysis..."
            echo "========================================="
            bash scripts/run_bucket_analysis_auto.sh
            exit 0
        else
            echo "[$(date +%H:%M:%S)] Job not in queue but log file not found. Job may have been cancelled."
            exit 1
        fi
    fi

    sleep $CHECK_INTERVAL
done
