#!/bin/bash
#
# Master script to submit all multi-seed experiments for TCVM and ALFAR
# Usage: bash submit_all_multiseed.sh [tcvm|alfar|all]
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse arguments
MODE=${1:-all}

if [[ ! "$MODE" =~ ^(tcvm|alfar|all)$ ]]; then
    print_error "Invalid argument: $MODE"
    echo "Usage: bash submit_all_multiseed.sh [tcvm|alfar|all]"
    exit 1
fi

# Change to slurm_jobs directory
cd /data/gpfs/projects/punim2075/ALFAR/slurm_jobs

print_header "Multi-Seed Experiment Submission"
echo "Mode: $MODE"
echo "Date: $(date)"
echo "User: $USER"
echo ""

# Arrays to store job IDs
declare -a TCVM_JOBS
declare -a ALFAR_JOBS

# Submit TCVM experiments
if [[ "$MODE" == "tcvm" || "$MODE" == "all" ]]; then
    print_header "Submitting TCVM Experiments (5 seeds each)"

    datasets=("okvqa" "aokvqa" "infoseek" "viquae" "evqa")

    for dataset in "${datasets[@]}"; do
        script="run_${dataset}_tcvm_multiseed.slurm"

        if [ ! -f "$script" ]; then
            print_warning "Script not found: $script, skipping..."
            continue
        fi

        echo -n "Submitting ${dataset} TCVM... "
        job_output=$(sbatch "$script" 2>&1)

        if [ $? -eq 0 ]; then
            job_id=$(echo "$job_output" | grep -oP '(?<=job )\d+')
            TCVM_JOBS+=("${dataset}:${job_id}")
            print_success "Job ID: $job_id"
        else
            print_error "Failed: $job_output"
        fi
    done

    echo ""
fi

# Submit ALFAR experiments
if [[ "$MODE" == "alfar" || "$MODE" == "all" ]]; then
    print_header "Submitting ALFAR Experiments (5 seeds each)"

    datasets=("okvqa" "aokvqa" "infoseek" "viquae" "evqa")

    for dataset in "${datasets[@]}"; do
        script="run_${dataset}_alfar_multiseed.slurm"

        if [ ! -f "$script" ]; then
            print_warning "Script not found: $script, skipping..."
            continue
        fi

        echo -n "Submitting ${dataset} ALFAR... "
        job_output=$(sbatch "$script" 2>&1)

        if [ $? -eq 0 ]; then
            job_id=$(echo "$job_output" | grep -oP '(?<=job )\d+')
            ALFAR_JOBS+=("${dataset}:${job_id}")
            print_success "Job ID: $job_id"
        else
            print_error "Failed: $job_output"
        fi
    done

    echo ""
fi

# Summary
print_header "Submission Summary"

if [[ "$MODE" == "tcvm" || "$MODE" == "all" ]]; then
    echo -e "${BLUE}TCVM Jobs:${NC}"
    if [ ${#TCVM_JOBS[@]} -eq 0 ]; then
        print_warning "No TCVM jobs submitted"
    else
        for job in "${TCVM_JOBS[@]}"; do
            dataset="${job%%:*}"
            job_id="${job##*:}"
            echo "  - ${dataset}: ${job_id} (array tasks: ${job_id}_0 to ${job_id}_4)"
        done
    fi
    echo ""
fi

if [[ "$MODE" == "alfar" || "$MODE" == "all" ]]; then
    echo -e "${BLUE}ALFAR Jobs:${NC}"
    if [ ${#ALFAR_JOBS[@]} -eq 0 ]; then
        print_warning "No ALFAR jobs submitted"
    else
        for job in "${ALFAR_JOBS[@]}"; do
            dataset="${job%%:*}"
            job_id="${job##*:}"
            echo "  - ${dataset}: ${job_id} (array tasks: ${job_id}_0 to ${job_id}_4)"
        done
    fi
    echo ""
fi

# Total count
total_jobs=$((${#TCVM_JOBS[@]} + ${#ALFAR_JOBS[@]}))
total_tasks=$((total_jobs * 5))  # Each job has 5 array tasks (seeds)

echo -e "${GREEN}Total Jobs: ${total_jobs} (${total_tasks} tasks)${NC}"
echo ""

# Monitoring commands
print_header "Monitoring Commands"
echo "Check all jobs:       squeue -u $USER"
echo "Check specific job:   squeue -j JOB_ID"
echo "View live log:        tail -f ../logs/tcvm_okvqa_seed0_JOBID.out"
echo "Cancel all jobs:      scancel -u $USER"
echo "Cancel specific job:  scancel JOB_ID"
echo ""

# Save job IDs to file
job_file="../logs/multiseed_jobs_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "Multi-Seed Jobs Submitted on $(date)"
    echo "=========================================="
    echo ""
    echo "TCVM Jobs:"
    for job in "${TCVM_JOBS[@]}"; do
        echo "  $job"
    done
    echo ""
    echo "ALFAR Jobs:"
    for job in "${ALFAR_JOBS[@]}"; do
        echo "  $job"
    done
} > "$job_file"

print_success "Job IDs saved to: $job_file"
echo ""

# Create a monitoring script
monitor_script="../scripts/monitor_multiseed_jobs.sh"
{
    echo "#!/bin/bash"
    echo "# Auto-generated monitoring script"
    echo ""
    echo "echo 'Job Status:'"
    echo "echo '========================================'"

    for job in "${TCVM_JOBS[@]}" "${ALFAR_JOBS[@]}"; do
        dataset="${job%%:*}"
        job_id="${job##*:}"
        echo "echo '${dataset}: Job ${job_id}'"
        echo "squeue -j ${job_id} --format='  Task %K: %T (%M)' 2>/dev/null || echo '  Completed or not found'"
    done

    echo "echo '========================================'"
} > "$monitor_script"
chmod +x "$monitor_script"

print_success "Monitoring script created: $monitor_script"
echo "  Run: bash $monitor_script"
echo ""

print_header "Next Steps"
echo "1. Monitor jobs: squeue -u $USER"
echo "2. Wait for all seeds to complete (~24 hours per dataset)"
echo "3. Aggregate results for each dataset:"
echo ""
echo "   # TCVM results"
for dataset in okvqa aokvqa infoseek viquae evqa; do
    echo "   python scripts/aggregate_multiseed_results.py --dataset $dataset --method tcvm"
done
echo ""
echo "   # ALFAR results"
for dataset in okvqa aokvqa infoseek viquae evqa; do
    echo "   python scripts/aggregate_multiseed_results.py --dataset $dataset --method alfar"
done
echo ""

print_success "All submissions complete!"
