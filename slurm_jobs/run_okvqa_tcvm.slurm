#!/bin/bash
#SBATCH --job-name=tcvm_okvqa
#SBATCH --account=punim2075
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/tcvm_okvqa_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/tcvm_okvqa_%j.err

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================="

# Set working directory
cd /data/gpfs/projects/punim2075/ALFAR

# Load required modules
module load CUDA/11.8.0

# Create logs directory if it doesn't exist
mkdir -p logs

# Create results directory if it doesn't exist
mkdir -p experiments/result

# Activate virtual environment
source ALFAR/bin/activate

# Verify environment
echo "Loaded modules:"
module list
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Navigate to experiments/eval directory
cd experiments/eval

# Run the evaluation script on OK-VQA dataset with TCVM
echo "Starting TCVM evaluation on OK-VQA dataset..."
echo "TCVM Configuration:"
echo "  - Top-K: 20 visual tokens"
echo "  - Alpha: 1.0 (contrastive penalty)"
echo "  - Beta: 0.7 (plausibility threshold)"
echo "  - Masking strategy: zero"
echo "========================================="

python alfar_okvqa_llava.py \
    --dataset okvqa \
    --image-folder ../../data/images/coco/val2014 \
    --model-path ../../models/llava-v1.5-7b \
    --answers-file ../result/okvqa_tcvm_results.csv \
    --temperature 1.0 \
    --top_p 1.0 \
    --cd_beta 0.7 \
    --att_alpha 0.4 \
    --use_tcvm \
    --tcvm_topk 20 \
    --tcvm_alpha 1.0 \
    --tcvm_beta 0.7 \
    --tcvm_mask_strategy zero \
    --seed 0

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Prediction completed successfully!"
    echo "Results saved to: experiments/result/okvqa_tcvm_results.csv"

    # Navigate back to project root
    cd /data/gpfs/projects/punim2075/ALFAR

    # Run evaluation
    echo "========================================="
    echo "Starting evaluation..."
    python evaluation/eval_okvqa.py --dataset okvqa

    echo "========================================="
    echo "All tasks completed successfully!"
    echo "End time: $(date)"
else
    echo "========================================="
    echo "Evaluation failed with error code: $?"
    echo "End time: $(date)"
    exit 1
fi
