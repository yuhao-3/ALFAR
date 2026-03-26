#!/bin/bash
# FIXED: Submit all TCVM-KAR multi-seed experiments with CORRECT image paths

echo "========================================="
echo "TCVM-KAR Multi-seed Job Submission (FIXED)"
echo "========================================="

# Fix and resubmit with correct paths
echo "Creating fixed SLURM scripts with correct image paths..."

# InfoSeek - FIXED PATH
cat > slurm_jobs/run_infoseek_tcvm_multiseed_v2.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=tcvm_info_ms
#SBATCH --output=logs/tcvm_infoseek_seed%a_%j.out
#SBATCH --error=logs/tcvm_infoseek_seed%a_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --array=1-2

source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
cd /data/gpfs/projects/punim2075/ALFAR

python experiments/eval/alfar_mc_llava.py \
    --dataset infoseek \
    --use_tcvm --tcvm_topk 20 --tcvm_alpha 1.0 --tcvm_beta 0.7 \
    --seed $SLURM_ARRAY_TASK_ID \
    --answers-file results/llava1.5/infoseek/infoseek_tcvm_seed${SLURM_ARRAY_TASK_ID}_results.jsonl \
    --image-folder data/images/infoseek_images \
    --model-path models/llava-v1.5-7b
EOF

# ViQuAE - FIXED PATH
cat > slurm_jobs/run_viquae_tcvm_multiseed_v2.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=tcvm_viqu_ms
#SBATCH --output=logs/tcvm_viquae_seed%a_%j.out
#SBATCH --error=logs/tcvm_viquae_seed%a_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --array=1-2

source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
cd /data/gpfs/projects/punim2075/ALFAR

python experiments/eval/alfar_mc_llava.py \
    --dataset viquae \
    --use_tcvm --tcvm_topk 20 --tcvm_alpha 1.0 --tcvm_beta 0.7 \
    --seed $SLURM_ARRAY_TASK_ID \
    --answers-file results/llava1.5/viquae/viquae_tcvm_seed${SLURM_ARRAY_TASK_ID}_results.jsonl \
    --image-folder data/images/viquae_images/images \
    --model-path models/llava-v1.5-7b
EOF

# A-OKVQA - FIXED PATH
cat > slurm_jobs/run_aokvqa_tcvm_multiseed_v2.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=tcvm_aokv_ms
#SBATCH --output=logs/tcvm_aokvqa_seed%a_%j.out
#SBATCH --error=logs/tcvm_aokvqa_seed%a_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --array=1-2

source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
cd /data/gpfs/projects/punim2075/ALFAR

python experiments/eval/alfar_okvqa_llava.py \
    --dataset aokvqa \
    --use_tcvm --tcvm_topk 20 --tcvm_alpha 1.0 --tcvm_beta 0.7 \
    --seed $SLURM_ARRAY_TASK_ID \
    --answers-file results/llava1.5/aokvqa/aokvqa_tcvm_seed${SLURM_ARRAY_TASK_ID}_results.csv \
    --image-folder data/images/coco/val2014 \
    --model-path models/llava-v1.5-7b
EOF

# OK-VQA - FIXED PATH
cat > slurm_jobs/run_okvqa_tcvm_multiseed_v2.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=tcvm_okvq_ms
#SBATCH --output=logs/tcvm_okvqa_seed%a_%j.out
#SBATCH --error=logs/tcvm_okvqa_seed%a_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --array=1-2

source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
cd /data/gpfs/projects/punim2075/ALFAR

python experiments/eval/alfar_okvqa_llava.py \
    --dataset okvqa \
    --use_tcvm --tcvm_topk 20 --tcvm_alpha 1.0 --tcvm_beta 0.7 \
    --seed $SLURM_ARRAY_TASK_ID \
    --answers-file results/llava1.5/okvqa/okvqa_tcvm_seed${SLURM_ARRAY_TASK_ID}_results.csv \
    --image-folder data/images/coco/val2014 \
    --model-path models/llava-v1.5-7b
EOF

# E-VQA - FIXED PATH
cat > slurm_jobs/run_evqa_tcvm_multiseed_v2.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=tcvm_evqa_ms
#SBATCH --output=logs/tcvm_evqa_seed%a_%j.out
#SBATCH --error=logs/tcvm_evqa_seed%a_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --array=1-2

source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
cd /data/gpfs/projects/punim2075/ALFAR

python experiments/eval/alfar_evqa_llava.py \
    --use_tcvm --tcvm_topk 20 --tcvm_alpha 1.0 --tcvm_beta 0.7 \
    --seed $SLURM_ARRAY_TASK_ID \
    --answers-file results/llava1.5/evqa/evqa_tcvm_seed${SLURM_ARRAY_TASK_ID}_results.json \
    --image-folder data/images/oven_eval \
    --model-path models/llava-v1.5-7b
EOF

echo "Fixed scripts created!"
echo "========================================="
echo "Submitting corrected jobs..."

JOB_INFO=$(sbatch slurm_jobs/run_infoseek_tcvm_multiseed_v2.slurm | awk '{print $4}')
JOB_VIQU=$(sbatch slurm_jobs/run_viquae_tcvm_multiseed_v2.slurm | awk '{print $4}')
JOB_AOKV=$(sbatch slurm_jobs/run_aokvqa_tcvm_multiseed_v2.slurm | awk '{print $4}')
JOB_OKVQ=$(sbatch slurm_jobs/run_okvqa_tcvm_multiseed_v2.slurm | awk '{print $4}')
JOB_EVQA=$(sbatch slurm_jobs/run_evqa_tcvm_multiseed_v2.slurm | awk '{print $4}')

echo "========================================="
echo "Summary - All Jobs Submitted!"
echo "========================================="
echo "  - InfoSeek: $JOB_INFO (seeds 1-2)"
echo "  - ViQuAE: $JOB_VIQU (seeds 1-2)"
echo "  - A-OKVQA: $JOB_AOKV (seeds 1-2)"
echo "  - OK-VQA: $JOB_OKVQ (seeds 1-2)"
echo "  - E-VQA: $JOB_EVQA (seeds 1-2)"
echo ""
echo "Total: 10 jobs (5 datasets × 2 seeds)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 10 'squeue -u \$USER'"
echo "========================================="
