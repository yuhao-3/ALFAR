# ALFAR & TCVM-KAR Troubleshooting Guide

Comprehensive guide for diagnosing and fixing common issues with ALFAR and TCVM-KAR experiments.

## Table of Contents
- [Environment Issues](#environment-issues)
- [SLURM Job Issues](#slurm-job-issues)
- [Python/Package Issues](#pythonpackage-issues)
- [Data Issues](#data-issues)
- [Model/CUDA Issues](#modelcuda-issues)
- [Evaluation Issues](#evaluation-issues)
- [Result Analysis Issues](#result-analysis-issues)

---

## Environment Issues

### Issue: NumPy 2.0 Incompatibility ✅ SOLVED

**Symptoms**:
```
RuntimeError: Could not infer dtype of numpy.float32
ValueError: Unable to create tensor
```

**Diagnosis**:
```bash
python3.9 -c "import numpy; print(numpy.__version__)"
# If output is 2.0.x or higher, this is the issue
```

**Solution**:
```bash
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
pip install "numpy<2.0" --upgrade
```

**Verification**:
```bash
python3.9 -c "import torch; import numpy as np; arr = np.array([1.0], dtype=np.float32); tensor = torch.tensor(arr); print('OK')"
```

**Expected**: Output should be `OK` without errors

---

### Issue: Virtual Environment Not Activated

**Symptoms**:
- `ModuleNotFoundError` for installed packages
- Wrong Python version
- Commands fail with "command not found"

**Diagnosis**:
```bash
which python3.9
# Should show: /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/python3.9

echo $VIRTUAL_ENV
# Should show: /data/gpfs/projects/punim2075/ALFAR/ALFAR
```

**Solution**:
```bash
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
```

**For SLURM jobs**: Ensure your `.slurm` script has:
```bash
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
```

---

### Issue: Missing Dependencies

**Symptoms**:
```
ModuleNotFoundError: No module named 'transformers'
ImportError: cannot import name 'xxx'
```

**Diagnosis**:
```bash
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
pip list | grep -i transformers
pip list | grep -i torch
```

**Solution**:
```bash
# Re-install requirements
pip install -r requirements.txt

# For specific package
pip install transformers --upgrade
```

---

## SLURM Job Issues

### Issue: Job Stuck in Pending (PD) State

**Symptoms**:
```bash
squeue -u $USER
# Shows: ST = PD, REASON = (Priority) or (Resources)
```

**Diagnosis**:
```bash
# Check detailed reason
squeue -u $USER --start

# Check partition status
sinfo -p gpu-a100
```

**Common Reasons**:
1. **(Priority)**: Other jobs have higher priority - wait your turn
2. **(Resources)**: No available GPUs - wait for resources
3. **(QOSMaxJobsPerUserLimit)**: You've hit job limit
4. **(ReqNodeNotAvail)**: Requested resources not available

**Solution**:
- **Wait**: Most jobs will start within 0-4 hours
- **Check limits**: `sacctmgr show qos format=name,maxjobsperuser`
- **Reduce resource request**: If asking for too much memory/time

**Monitor**:
```bash
watch -n 60 'squeue -u $USER'
```

---

### Issue: Job Failed Immediately

**Symptoms**:
```bash
sacct -u $USER --format=JobID,JobName,State,Elapsed
# Shows: State = FAILED, Elapsed = 00:00:01
```

**Diagnosis**:
```bash
# Find the job ID
sacct -u $USER --format=JobID,JobName,State,Start

# Check error log
cat logs/[jobname]_[jobid].err

# Check output log
tail -50 logs/[jobname]_[jobid].out
```

**Common Causes**:
1. **Syntax error in SLURM script**: Check shebang, module loads
2. **Virtual environment not activated**: Add `source .../activate`
3. **Python script not found**: Check paths
4. **Permission denied**: Check file permissions with `ls -l`

**Solution**:
```bash
# Validate SLURM script syntax
bash -n slurm_jobs/run_dataset_method.slurm

# Test Python script locally
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
python experiments/eval/[script].py --help
```

---

### Issue: Job Killed by SLURM

**Symptoms**:
```bash
sacct -j [JOBID]
# Shows: State = CANCELLED or TIMEOUT or OUT_OF_MEMORY
```

**Diagnosis**:
```bash
# Check why job was killed
sacct -j [JOBID] --format=JobID,State,ExitCode,MaxRSS,Elapsed,Timelimit

# Check for OOM
grep -i "out of memory\|oom\|killed" logs/[jobname]_[jobid].err
```

**Solutions**:

**TIMEOUT**:
```bash
# Increase time limit in .slurm file
#SBATCH --time=48:00:00  # Increase from 24 to 48 hours
```

**OUT_OF_MEMORY (CPU)**:
```bash
# Increase RAM in .slurm file
#SBATCH --mem=128GB  # Increase from 64GB to 128GB
```

**OUT_OF_MEMORY (GPU)**:
```bash
# Check CUDA OOM in error log
grep "CUDA out of memory" logs/[jobname]_[jobid].err

# Solutions:
# 1. Reduce batch size in script
# 2. Use gradient checkpointing (if applicable)
# 3. Request A100-80GB instead of A100-40GB
```

---

### Issue: Can't Cancel Job

**Symptoms**:
```bash
scancel [JOBID]
# Job still shows in squeue
```

**Solution**:
```bash
# Force cancel
scancel --signal=KILL [JOBID]

# Cancel all your jobs
scancel -u $USER

# If still stuck, wait 1-2 minutes and check again
squeue -u $USER
```

---

## Python/Package Issues

### Issue: Import Errors for Custom Modules

**Symptoms**:
```
ModuleNotFoundError: No module named 'experiments'
ImportError: cannot import name 'attention'
```

**Diagnosis**:
```bash
echo $PYTHONPATH
# Check if /data/gpfs/projects/punim2075/ALFAR is in the path

pwd
# Check you're in the correct directory
```

**Solution**:
```bash
# Method 1: Set PYTHONPATH
export PYTHONPATH=/data/gpfs/projects/punim2075/ALFAR:$PYTHONPATH

# Method 2: Run from correct directory
cd /data/gpfs/projects/punim2075/ALFAR
python experiments/eval/script.py

# For SLURM: Add to .slurm script
export PYTHONPATH=/data/gpfs/projects/punim2075/ALFAR:$PYTHONPATH
```

---

### Issue: Transformers/Model Loading Errors

**Symptoms**:
```
OSError: Can't load weights for 'liuhaotian/llava-v1.5-7b'
ValueError: Checkpoint format not recognized
```

**Diagnosis**:
```bash
# Check if model exists locally
ls -lh models/llava-v1.5-7b/

# Check HuggingFace cache
ls -lh ~/.cache/huggingface/hub/
```

**Solution**:
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b
python -c "from transformers import AutoModel; AutoModel.from_pretrained('liuhaotian/llava-v1.5-7b')"

# Or use local path in script
--model-path /data/gpfs/projects/punim2075/ALFAR/models/llava-v1.5-7b
```

---

## Data Issues

### Issue: Image Files Not Found

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/image.jpg'
PIL.UnidentifiedImageError: cannot identify image file
```

**Diagnosis**:
```bash
# Check if image directory exists
ls -ld data/images/[dataset]/

# Check image count
find data/images/[dataset]/ -name "*.jpg" | wc -l
find data/images/[dataset]/ -name "*.JPEG" | wc -l

# Check specific image
ls -lh data/images/[dataset]/[image_id].jpg
```

**Solution**:
```bash
# Download images if missing
# InfoSeek
./scripts/download_infoseek_images.sh

# COCO (for OKVQA/AOKVQA)
./scripts/download_coco.sh

# Check image permissions
chmod -R u+r data/images/[dataset]/
```

---

### Issue: Knowledge Base Not Found

**Symptoms**:
```
FileNotFoundError: data/wiki/[file] not found
ValueError: Retrieved knowledge index out of range
```

**Diagnosis**:
```bash
# Check wiki directory
ls -lh data/wiki/

# Check retrieval result files
ls -lh data/retrieval_result/
```

**Solution**:
```bash
# Download knowledge base from Google Drive
# See main README.md for link

# Verify file structure
data/
├── wiki/
│   ├── [knowledge files]
└── retrieval_result/
    ├── aokvqa_retrieval.json
    ├── infoseek_retrieval.json
    └── ...
```

---

### Issue: Retrieval Results Malformed

**Symptoms**:
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
KeyError: 'question_id' or 'retrieved_docs'
```

**Diagnosis**:
```bash
# Check file is valid JSON
python3.9 -c "import json; json.load(open('data/retrieval_result/aokvqa_retrieval.json'))"

# Check first few lines
head -5 data/retrieval_result/aokvqa_retrieval.json
```

**Solution**:
```bash
# Re-download retrieval results
# Or regenerate if you have the retrieval script

# Validate JSON structure
python3.9 -c "
import json
data = json.load(open('data/retrieval_result/aokvqa_retrieval.json'))
print(f'Loaded {len(data)} entries')
print(f'Keys: {list(data[0].keys())}')
"
```

---

## Model/CUDA Issues

### Issue: CUDA Not Available

**Symptoms**:
```
RuntimeError: No CUDA GPUs are available
torch.cuda.is_available() returns False
```

**Diagnosis**:
```bash
# Check CUDA module
module list | grep -i cuda

# Check GPU allocation in SLURM
echo $CUDA_VISIBLE_DEVICES

# Test CUDA
python3.9 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}')"
```

**Solution**:
```bash
# For interactive session: Load CUDA module
module load CUDA/11.8.0

# For SLURM job: Ensure in .slurm file
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100

# Check GPU allocation
nvidia-smi  # Should show GPU info
```

---

### Issue: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
torch.cuda.OutOfMemoryError
```

**Diagnosis**:
```bash
# Check GPU memory usage
nvidia-smi

# Check model size
grep "model size\|parameters" logs/[jobname]_[jobid].out
```

**Solution**:
```python
# In evaluation script, reduce batch processing
# For LLaVA, it processes one image at a time, so:

# 1. Use half precision (already done)
images=raw_image_tensor.unsqueeze(0).half().cuda()

# 2. Clear cache periodically
if idx % 100 == 0:
    torch.cuda.empty_cache()

# 3. Use smaller model or request A100-80GB
#SBATCH --partition=gpu-a100-80gb  # If available
```

---

### Issue: Model Weights Mismatch

**Symptoms**:
```
RuntimeError: Error(s) in loading state_dict for LlavaLlamaForCausalLM:
    size mismatch for model.layers.0.self_attn.q_proj.weight
```

**Diagnosis**:
```bash
# Check model path
ls -lh models/llava-v1.5-7b/

# Verify config
cat models/llava-v1.5-7b/config.json
```

**Solution**:
```bash
# Re-download model
rm -rf models/llava-v1.5-7b/
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('liuhaotian/llava-v1.5-7b')
model.save_pretrained('models/llava-v1.5-7b/')
"

# Or use HuggingFace path directly
--model-path liuhaotian/llava-v1.5-7b
```

---

## Evaluation Issues

### Issue: Empty Result Files

**Symptoms**:
```bash
ls -lh experiments/result/*_results.*
# Shows: 0 bytes for result files
```

**Diagnosis**:
```bash
# Check if evaluation completed
tail -50 logs/[jobname]_[jobid].out

# Look for errors
grep -i "error\|exception\|failed" logs/[jobname]_[jobid].out
```

**Common Causes**:
1. **Job failed early**: Check error log
2. **Wrong output path**: Check script `--output` argument
3. **Permission issue**: Check directory permissions

**Solution**:
```bash
# Check script output path
grep "output\|save" experiments/eval/[script].py

# Ensure output directory exists
mkdir -p experiments/result/

# Check permissions
ls -ld experiments/result/
chmod u+w experiments/result/
```

---

### Issue: Evaluation Script Errors

**Symptoms**:
```
ValueError: Accuracy calculation failed
KeyError: 'answer' or 'prediction'
```

**Diagnosis**:
```bash
# Check result file format
head -5 experiments/result/[dataset]_[method]_results.csv

# Verify with Python
python3.9 -c "
import pandas as pd
df = pd.read_csv('experiments/result/aokvqa_alfar.csv')
print(df.columns)
print(df.head())
"
```

**Solution**:
```bash
# For OKVQA/AOKVQA (CSV format)
# Expected columns: question_id, answer, prediction

# For InfoSeek/ViQuAE (JSONL format)
# Expected fields: question_id, answer, prediction

# For EVQA (JSON format)
# Expected structure: dict with question_id keys

# Regenerate results if format is wrong
```

---

## Result Analysis Issues

### Issue: Missing Metrics

**Symptoms**:
```bash
cat logs/[method]_[dataset]_seed0_metrics.txt
# File empty or shows errors
```

**Diagnosis**:
```bash
# Check if evaluation ran
ls -lh logs/*metrics.txt

# Check for evaluation errors
grep "Accuracy\|Error" logs/[method]_[dataset]_*.txt
```

**Solution**:
```bash
# Re-run evaluation
python evaluation/eval_okvqa.py --dataset aokvqa --preds experiments/result/aokvqa_alfar.csv > logs/alfar_aokvqa_metrics.txt

# Use batch script
bash scripts/generate_missing_metrics.sh
```

---

### Issue: Multi-seed Results Incomplete

**Symptoms**:
```bash
ls experiments/result/*seed*.csv
# Missing some seed files (should have seed0-4)
```

**Diagnosis**:
```bash
# Check which seeds completed
for seed in 0 1 2 3 4; do
    echo "Seed $seed:"
    ls -lh experiments/result/*seed$seed* 2>/dev/null || echo "  MISSING"
done
```

**Solution**:
```bash
# Resubmit missing seeds
sbatch slurm_jobs/run_aokvqa_alfar_multiseed.slurm  # Submits all 5 seeds

# Or submit specific seed
sbatch --export=SEED=3 slurm_jobs/run_aokvqa_alfar_seed.slurm
```

---

## Quick Diagnostic Commands

### Check Everything
```bash
# Environment
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
python3.9 -c "import torch, numpy as np, transformers; print('OK')"

# Jobs
squeue -u $USER
sacct -u $USER --starttime=2026-03-24 --format=JobID,JobName,State,Elapsed

# Results
ls -lh experiments/result/*tcvm* experiments/result/*alfar*

# Logs
ls -lht logs/*.out | head -10

# Data
ls -ld data/images/* data/wiki/ data/retrieval_result/
```

### Common Error Patterns
```bash
# Find all errors in logs
grep -i "error\|exception\|failed" logs/*.out logs/*.err

# Find CUDA errors
grep -i "cuda\|gpu\|out of memory" logs/*.err

# Find import errors
grep -i "modulenotfound\|importerror" logs/*.err

# Find file not found errors
grep -i "filenotfound\|no such file" logs/*.err
```

---

## Getting Help

If issues persist after trying these solutions:

1. **Check Documentation**:
   - `docs/TCVM_KAR_QUICK_REFERENCE.md`
   - `docs/TCVM_KAR_RUNTIME_STATUS.md`
   - `docs/ALFAR_TECHNICAL_DOCUMENTATION.md`

2. **Verify Environment**:
   ```bash
   source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
   python3.9 --version
   pip list | grep -E "torch|numpy|transformers"
   ```

3. **Check SLURM Documentation**:
   ```bash
   man sbatch
   man squeue
   man scancel
   ```

4. **Collect Diagnostic Information**:
   - Job ID
   - Error logs (`logs/[jobname]_[jobid].err`)
   - Last 50 lines of output (`tail -50 logs/[jobname]_[jobid].out`)
   - Environment info (`pip list`, `module list`)

---

**Last Updated**: March 25, 2026
**Applies to**: ALFAR, TCVM, TCVM-KAR experiments on Spartan HPC
