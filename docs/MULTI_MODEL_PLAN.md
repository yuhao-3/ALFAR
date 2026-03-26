# Multi-Model TCVM-KAR Experiment Plan

**Created**: March 25, 2026
**Goal**: Run TCVM-KAR on multiple vision-language models
**Priority Datasets**: InfoSeek, ViQuAE

---

## Quick Summary

✅ **Completed**: LLaVA 1.5-7B on all 5 datasets
⏳ **Next**: InstructBLIP, Shikra, MiniGPT-4, LLaVA-NEXT, Qwen2.5-VL on InfoSeek & ViQuAE

---

## Results Organization

All results are saved in: `results/{model}/{dataset}/`

```
results/
├── RESULTS_SUMMARY.md          # Master summary document
└── llava1.5/                    # ✅ COMPLETED
    ├── infoseek/
    │   ├── infoseek_tcvm_results.jsonl
    │   └── accuracy.txt (57.23%)
    ├── viquae/
    │   ├── viquae_tcvm_results.jsonl
    │   └── accuracy.txt (57.07%)
    ├── aokvqa/
    │   ├── aokvqa_tcvm_results.csv
    │   └── accuracy.txt (59.71%)
    ├── okvqa/
    │   ├── okvqa_tcvm_results.csv
    │   └── accuracy.txt (60.66%)
    └── evqa/
        ├── evqa_tcvm_results.json
        └── accuracy.txt (35.97%)
```

---

## Models to Run (Priority: InfoSeek & ViQuAE)

### 1. InstructBLIP
- **Script**: `experiments/eval/alfar_mc_instructblip.py`
- **Status**: ⏳ Needs TCVM-KAR integration
- **Model**: BLIP-2 Vicuna Instruct 7B
- **Datasets**: InfoSeek, ViQuAE
- **Action Required**: Modify script to add TCVM parameters

### 2. Shikra
- **Script**: `experiments/eval/alfar_mc_shikra.py`
- **Status**: ⏳ Needs TCVM-KAR integration
- **Model**: Shikra 7B
- **Datasets**: InfoSeek, ViQuAE
- **Action Required**: Modify script to add TCVM parameters

### 3. MiniGPT-4
- **Script**: `experiments/eval/alfar_mc_minigpt.py`
- **Status**: ⏳ Needs TCVM-KAR integration
- **Model**: MiniGPT-4 (Llama 2 Chat 7B)
- **Datasets**: InfoSeek, ViQuAE
- **Action Required**: Modify script to add TCVM parameters

### 4. LLaVA-NEXT (LLaVA 1.6)
- **Script**: Need to create
- **Status**: ⏳ New script required
- **Model**: LLaVA-NEXT
- **Datasets**: InfoSeek, ViQuAE
- **Action Required**: Create new script based on LLaVA 1.5 structure

### 5. Qwen2.5-VL
- **Script**: Need to create
- **Status**: ⏳ New script required
- **Model**: Qwen2.5-VL
- **Datasets**: InfoSeek, ViQuAE
- **Action Required**: Create new script or adapter

---

## Implementation Steps

### Step 1: Modify Existing Scripts (InstructBLIP, Shikra, MiniGPT-4)

Each script needs three changes:

#### Change 1: Add TCVM arguments to parser

```python
# Add after existing arguments
parser.add_argument("--use_tcvm", action="store_true",
                    help="Enable TCVM-KAR decoding")
parser.add_argument("--tcvm_topk", type=int, default=20,
                    help="Number of tokens to mask")
parser.add_argument("--tcvm_alpha", type=float, default=1.0,
                    help="Contrastive penalty weight")
parser.add_argument("--tcvm_beta", type=float, default=0.7,
                    help="Plausibility threshold for APC")
parser.add_argument("--tcvm_mask_strategy", type=str, default='zero',
                    choices=['zero', 'random'], help="Token masking strategy")
```

#### Change 2: Pass TCVM parameters to generate()

```python
# In the generation call
output = model.generate(
    # ... existing parameters ...
    use_tcvm=args.use_tcvm,
    tcvm_topk=args.tcvm_topk,
    tcvm_alpha=args.tcvm_alpha,
    tcvm_beta=args.tcvm_beta,
    tcvm_mask_strategy=args.tcvm_mask_strategy,
    # ... rest of parameters ...
)
```

#### Change 3: Update output path

```python
# Change answers_file to use new results structure
answers_file = f"results/{model_name}/{args.dataset}/{args.dataset}_tcvm_results.jsonl"
```

**Note**: The `vcd_sample.evolve_vcd_sampling()` is already called in these scripts, so TCVM support is available once parameters are passed.

### Step 2: Create SLURM Job Scripts

For each model-dataset combination:

**File**: `slurm_jobs/run_{dataset}_{model}_tcvm.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=tcvm_{model}_{dataset}
#SBATCH --output=logs/tcvm_{dataset}_{model}_%j.out
#SBATCH --error=logs/tcvm_{dataset}_{model}_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================="

# Activate environment
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate

# Set working directory
cd /data/gpfs/projects/punim2075/ALFAR

# Run TCVM-KAR evaluation
python experiments/eval/alfar_mc_{model}.py \
    --dataset {dataset} \
    --use_tcvm \
    --tcvm_topk 20 \
    --tcvm_alpha 1.0 \
    --tcvm_beta 0.7 \
    --tcvm_mask_strategy zero \
    --seed 0 \
    --answers-file results/{model}/{dataset}/{dataset}_tcvm_results.jsonl \
    --image-folder data/images/{dataset}/

echo "========================================="
echo "End time: $(date)"
```

### Step 3: Create Batch Submission Script

**File**: `slurm_jobs/run_all_models_infoseek_viquae.sh`

```bash
#!/bin/bash
# Submit TCVM-KAR experiments for all models on InfoSeek and ViQuAE

MODELS=("instructblip" "shikra" "minigpt4")
DATASETS=("infoseek" "viquae")

echo "Submitting TCVM-KAR experiments for InfoSeek and ViQuAE"
echo "======================================================="

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Submitting: $model on $dataset"
        sbatch slurm_jobs/run_${dataset}_${model}_tcvm.slurm
    done
done

echo "======================================================="
echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
```

---

## Execution Plan

### Phase 1: InstructBLIP (Week 1)

**Tasks**:
1. Modify `experiments/eval/alfar_mc_instructblip.py`
2. Create SLURM scripts for InfoSeek and ViQuAE
3. Submit jobs
4. Monitor and evaluate results
5. Save to `results/instructblip/`

**Commands**:
```bash
# After modifications
sbatch slurm_jobs/run_infoseek_instructblip_tcvm.slurm
sbatch slurm_jobs/run_viquae_instructblip_tcvm.slurm

# Evaluate when complete
python evaluation/eval_mc.py --dataset infoseek --preds results/instructblip/infoseek/infoseek_tcvm_results.jsonl
python evaluation/eval_mc.py --dataset viquae --preds results/instructblip/viquae/viquae_tcvm_results.jsonl
```

### Phase 2: Shikra (Week 2)

**Tasks**:
1. Modify `experiments/eval/alfar_mc_shikra.py`
2. Create SLURM scripts
3. Submit and evaluate

### Phase 3: MiniGPT-4 (Week 3)

**Tasks**:
1. Modify `experiments/eval/alfar_mc_minigpt.py`
2. Create SLURM scripts
3. Submit and evaluate

### Phase 4: LLaVA-NEXT (Week 4)

**Tasks**:
1. Create new script based on LLaVA 1.5
2. Test on small subset
3. Submit and evaluate

### Phase 5: Qwen2.5-VL (Week 5)

**Tasks**:
1. Create new script or adapter
2. Test on small subset
3. Submit and evaluate

---

## Expected Results Table

After all experiments complete:

| Model | InfoSeek | ViQuAE | Notes |
|-------|----------|---------|-------|
| **LLaVA 1.5** | 57.23% ✅ | 57.07% ✅ | Baseline complete |
| InstructBLIP | TBD | TBD | BLIP-2 architecture |
| Shikra | TBD | TBD | Referring expression support |
| MiniGPT-4 | TBD | TBD | Llama 2 based |
| LLaVA-NEXT | TBD | TBD | LLaVA 1.6 improvements |
| Qwen2.5-VL | TBD | TBD | Latest Qwen VL model |

---

## Configuration (All Models)

Consistent TCVM-KAR configuration for fair comparison:

```python
use_tcvm = True
tcvm_topk = 20              # Top-K tokens to mask
tcvm_alpha = 1.0            # Contrastive penalty weight
tcvm_beta = 0.7             # Plausibility threshold (APC)
tcvm_mask_strategy = 'zero' # Zero-out masked tokens
seed = 0                    # Random seed for reproducibility
```

---

## Monitoring

### Check Job Status
```bash
squeue -u $USER
```

### Monitor All Running Jobs
```bash
watch -n 10 'squeue -u $USER'
```

### Check Results
```bash
# View results summary
cat results/RESULTS_SUMMARY.md

# Check specific model results
ls -lh results/instructblip/infoseek/
cat results/instructblip/infoseek/accuracy.txt
```

---

## Timeline Estimate

| Week | Model | Datasets | Status |
|------|-------|----------|--------|
| 0 (Complete) | LLaVA 1.5 | All 5 | ✅ Done |
| 1 | InstructBLIP | InfoSeek, ViQuAE | ⏳ Pending |
| 2 | Shikra | InfoSeek, ViQuAE | ⏳ Pending |
| 3 | MiniGPT-4 | InfoSeek, ViQuAE | ⏳ Pending |
| 4 | LLaVA-NEXT | InfoSeek, ViQuAE | ⏳ Pending |
| 5 | Qwen2.5-VL | InfoSeek, ViQuAE | ⏳ Pending |

**Total**: ~5 weeks for all models

---

## Next Immediate Steps

1. **Modify InstructBLIP script**: Add TCVM-KAR support to `alfar_mc_instructblip.py`
2. **Create SLURM scripts**: For InstructBLIP InfoSeek and ViQuAE
3. **Test on subset**: Verify modifications work correctly
4. **Submit jobs**: Start InstructBLIP experiments
5. **Repeat**: For Shikra, MiniGPT-4, etc.

---

## File Checklist

Before running each model:

- [ ] Modified evaluation script with TCVM parameters
- [ ] Created SLURM job scripts (one per dataset)
- [ ] Created results folder: `results/{model}/{dataset}/`
- [ ] Tested on small subset first
- [ ] Verified model weights are available
- [ ] Checked data paths are correct

---

## References

- **TCVM-KAR Implementation**: `experiments/eval/vcd_sample.py`
- **Example (LLaVA)**: `experiments/eval/alfar_mc_llava.py`
- **Evaluation**: `evaluation/eval_mc.py`
- **Results Summary**: `results/RESULTS_SUMMARY.md`
- **Complete Guide**: `docs/TCVM_KAR_RESULTS.md`

---

**Version**: 1.0
**Last Updated**: March 25, 2026
**Status**: Planning phase complete, ready for implementation
