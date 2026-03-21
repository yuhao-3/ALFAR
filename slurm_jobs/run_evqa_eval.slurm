#!/bin/bash
#SBATCH --job-name=alfar_evqa
#SBATCH --account=punim2075
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/alfar_evqa_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/alfar_evqa_%j.err

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

# Add custom Python libraries to PYTHONPATH
export PYTHONPATH=/data/gpfs/projects/punim2075/ALFAR/python_libs:$PYTHONPATH

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

# Run the evaluation script on E-VQA dataset
echo "Starting ALFAR evaluation on E-VQA dataset..."
python alfar_evqa_llava.py \
    --image-folder ../../data/images/iNaturalist2021 \
    --model-path ../../models/llava-v1.5-7b \
    --answers-file ../result/evqa_alfar_results.jsonl \
    --temperature 1.0 \
    --top_p 1.0 \
    --cd_beta 0.7 \
    --att_alpha 0.1 \
    --seed 0

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Prediction completed successfully!"
    echo "Results saved to: experiments/result/evqa_alfar_results.jsonl"

    # Navigate back to project root
    cd /data/gpfs/projects/punim2075/ALFAR

    # Run evaluation
    echo "========================================="
    echo "Starting evaluation..."
    python evaluation/eval_evqa.py \
        --input_file experiments/result/evqa_alfar_results.jsonl

    echo "========================================="
    echo "All tasks completed successfully!"
    echo "End time: $(date)"
else
    echo "========================================="
    echo "Evaluation failed with error code: $?"
    echo "End time: $(date)"
    exit 1
fi
