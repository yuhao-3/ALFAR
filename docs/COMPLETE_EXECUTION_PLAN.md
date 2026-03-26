# Complete Multi-Model Multi-Seed Execution Plan

**Created**: March 25, 2026, 21:15 AEDT
**Goal**: Run 3 seeds for each model-dataset combination
**Format**: Report all results as Mean(SD)

---

## 🎯 MASTER PLAN OVERVIEW

### Total Scope
- **Models**: 6 (LLaVA 1.5, InstructBLIP, Shikra, MiniGPT-4, LLaVA-NEXT, Qwen2.5-VL)
- **Priority Datasets**: InfoSeek, ViQuAE (focus datasets)
- **All Datasets**: InfoSeek, ViQuAE, A-OKVQA, OK-VQA, E-VQA
- **Seeds per experiment**: 3 (0, 1, 2)
- **Total experiments**: 36 core + additional

### Expected Timeline
- **LLaVA 1.5 (all 5 datasets)**: 2-3 days (RUNNING NOW)
- **Other models (2 datasets each)**: ~1 week per model
- **Total**: ~6 weeks for complete results

---

## 📊 PHASE 1: LLaVA 1.5 (CURRENTLY RUNNING) ✅

### Status: ACTIVE
**Start**: March 25, 2026, 21:00 AEDT
**Expected Completion**: March 27, 2026

### Jobs Running/Queued

| Job Array | Dataset   | Seeds | Status          | Progress |
|-----------|-----------|-------|-----------------|----------|
| 23175470  | InfoSeek  | 1-2   | 1 RUN, 1 PEND  | Seed 0 ✅ |
| 23175471  | ViQuAE    | 1-2   | PENDING         | Seed 0 ✅ |
| 23175472  | A-OKVQA   | 1-2   | PENDING         | Seed 0 ✅ |
| 23175473  | OK-VQA    | 1-2   | PENDING         | Seed 0 ✅ |
| 23175474  | E-VQA     | 1-2   | PENDING         | Seed 0 ✅ |

**Note**: Seed 0 already completed, using existing results from earlier today.

### Seed 0 Baseline Results

| Dataset   | Accuracy  | Status |
|-----------|-----------|--------|
| InfoSeek  | 57.23%    | ✅ Complete |
| ViQuAE    | 57.07%    | ✅ Complete |
| A-OKVQA   | 59.71%    | ✅ Complete |
| OK-VQA    | 60.66%    | ✅ Complete |
| E-VQA     | 35.97%    | ✅ Complete |

### Action After Phase 1 Completion
```bash
# Aggregate all LLaVA 1.5 results
python scripts/aggregate_tcvm_multiseed.py --model llava1.5 --all

# Expected output format:
# Dataset    | Seed 0  | Seed 1  | Seed 2  | Mean(SD)
# InfoSeek   | 57.23%  | XX.XX%  | XX.XX%  | XX.XX%(YY.YY%)
# ViQuAE     | 57.07%  | XX.XX%  | XX.XX%  | XX.XX%(YY.YY%)
# A-OKVQA    | 59.71%  | XX.XX%  | XX.XX%  | XX.XX%(YY.YY%)
# OK-VQA     | 60.66%  | XX.XX%  | XX.XX%  | XX.XX%(YY.YY%)
# E-VQA      | 35.97%  | XX.XX%  | XX.XX%  | XX.XX%(YY.YY%)
```

---

## 📊 PHASE 2: InstructBLIP (InfoSeek, ViQuAE)

### Status: READY TO START
**Start After**: LLaVA 1.5 complete
**Estimated Duration**: 3-4 days

### Prerequisites
1. ✅ Script exists: `experiments/eval/alfar_mc_instructblip.py`
2. ⏳ Need to add TCVM-KAR support
3. ⏳ Create multi-seed SLURM scripts

### Implementation Steps

#### Step 1: Modify InstructBLIP Script
```bash
# Add TCVM parameters to alfar_mc_instructblip.py
# Location: experiments/eval/alfar_mc_instructblip.py
```

**Required changes**:
```python
# Add arguments (around line 140)
parser.add_argument("--use_tcvm", action="store_true")
parser.add_argument("--tcvm_topk", type=int, default=20)
parser.add_argument("--tcvm_alpha", type=float, default=1.0)
parser.add_argument("--tcvm_beta", type=float, default=0.7)
parser.add_argument("--tcvm_mask_strategy", type=str, default='zero')

# Pass to generate() (around line 115)
output = model.generate(
    # ... existing params ...
    use_tcvm=args.use_tcvm,
    tcvm_topk=args.tcvm_topk,
    tcvm_alpha=args.tcvm_alpha,
    tcvm_beta=args.tcvm_beta,
    tcvm_mask_strategy=args.tcvm_mask_strategy,
)
```

#### Step 2: Create Multi-Seed SLURM Scripts
```bash
# Create: slurm_jobs/run_infoseek_instructblip_multiseed.slurm
# Create: slurm_jobs/run_viquae_instructblip_multiseed.slurm
```

**Template**:
```bash
#!/bin/bash
#SBATCH --job-name=tcvm_iblip_info
#SBATCH --array=0-2
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

python experiments/eval/alfar_mc_instructblip.py \
    --dataset infoseek \
    --use_tcvm --tcvm_topk 20 --tcvm_alpha 1.0 --tcvm_beta 0.7 \
    --seed $SLURM_ARRAY_TASK_ID \
    --answers-file results/instructblip/infoseek/infoseek_tcvm_seed${SLURM_ARRAY_TASK_ID}_results.jsonl
```

#### Step 3: Submit Jobs
```bash
mkdir -p results/instructblip/{infoseek,viquae}
sbatch slurm_jobs/run_infoseek_instructblip_multiseed.slurm
sbatch slurm_jobs/run_viquae_instructblip_multiseed.slurm
```

#### Step 4: Aggregate Results
```bash
python scripts/aggregate_tcvm_multiseed.py --model instructblip --dataset infoseek
python scripts/aggregate_tcvm_multiseed.py --model instructblip --dataset viquae
```

### Expected Results

| Dataset   | Seed 0 | Seed 1 | Seed 2 | Mean(SD) |
|-----------|--------|--------|--------|----------|
| InfoSeek  | TBD    | TBD    | TBD    | TBD      |
| ViQuAE    | TBD    | TBD    | TBD    | TBD      |

---

## 📊 PHASE 3: Shikra (InfoSeek, ViQuAE)

### Status: READY TO START
**Start After**: InstructBLIP complete
**Estimated Duration**: 3-4 days

### Prerequisites
1. ✅ Script exists: `experiments/eval/alfar_mc_shikra.py`
2. ⏳ Need to add TCVM-KAR support
3. ⏳ Create multi-seed SLURM scripts

### Implementation Steps
Same as InstructBLIP, but for Shikra:
1. Modify `alfar_mc_shikra.py`
2. Create SLURM scripts
3. Submit 6 jobs (3 seeds × 2 datasets)
4. Aggregate results

---

## 📊 PHASE 4: MiniGPT-4 (InfoSeek, ViQuAE)

### Status: READY TO START
**Start After**: Shikra complete
**Estimated Duration**: 3-4 days

### Prerequisites
1. ✅ Script exists: `experiments/eval/alfar_mc_minigpt.py`
2. ⏳ Need to add TCVM-KAR support
3. ⏳ Create multi-seed SLURM scripts

### Implementation Steps
Same as InstructBLIP, but for MiniGPT-4:
1. Modify `alfar_mc_minigpt.py`
2. Create SLURM scripts
3. Submit 6 jobs (3 seeds × 2 datasets)
4. Aggregate results

---

## 📊 PHASE 5: LLaVA-NEXT (InfoSeek, ViQuAE)

### Status: NEEDS SCRIPT CREATION
**Start After**: MiniGPT-4 complete
**Estimated Duration**: 4-5 days (includes script creation)

### Prerequisites
1. ⏳ Create new script based on LLaVA 1.5
2. ⏳ Test on small subset
3. ⏳ Create multi-seed SLURM scripts

### Implementation Steps
1. Create `alfar_mc_llava_next.py` based on `alfar_mc_llava.py`
2. Adapt for LLaVA-NEXT architecture
3. Add TCVM-KAR support
4. Test on 10 examples
5. Create SLURM scripts
6. Submit 6 jobs (3 seeds × 2 datasets)
7. Aggregate results

---

## 📊 PHASE 6: Qwen2.5-VL (InfoSeek, ViQuAE)

### Status: NEEDS SCRIPT CREATION
**Start After**: LLaVA-NEXT complete
**Estimated Duration**: 4-5 days (includes script creation)

### Prerequisites
1. ⏳ Create new script for Qwen2.5-VL
2. ⏳ Install Qwen2.5-VL dependencies
3. ⏳ Test on small subset
4. ⏳ Create multi-seed SLURM scripts

### Implementation Steps
1. Research Qwen2.5-VL integration
2. Create evaluation script
3. Add TCVM-KAR support
4. Test on 10 examples
5. Create SLURM scripts
6. Submit 6 jobs (3 seeds × 2 datasets)
7. Aggregate results

---

## 📈 EXPECTED FINAL RESULTS TABLE

After all phases complete, we'll have:

### InfoSeek Results (Mean ± SD)

| Model         | Seed 0  | Seed 1 | Seed 2 | **Mean(SD)** |
|---------------|---------|--------|--------|--------------|
| LLaVA 1.5     | 57.23%  | ⏳     | ⏳     | ⏳           |
| InstructBLIP  | ⏳      | ⏳     | ⏳     | ⏳           |
| Shikra        | ⏳      | ⏳     | ⏳     | ⏳           |
| MiniGPT-4     | ⏳      | ⏳     | ⏳     | ⏳           |
| LLaVA-NEXT    | ⏳      | ⏳     | ⏳     | ⏳           |
| Qwen2.5-VL    | ⏳      | ⏳     | ⏳     | ⏳           |

### ViQuAE Results (Mean ± SD)

| Model         | Seed 0  | Seed 1 | Seed 2 | **Mean(SD)** |
|---------------|---------|--------|--------|--------------|
| LLaVA 1.5     | 57.07%  | ⏳     | ⏳     | ⏳           |
| InstructBLIP  | ⏳      | ⏳     | ⏳     | ⏳           |
| Shikra        | ⏳      | ⏳     | ⏳     | ⏳           |
| MiniGPT-4     | ⏳      | ⏳     | ⏳     | ⏳           |
| LLaVA-NEXT    | ⏳      | ⏳     | ⏳     | ⏳           |
| Qwen2.5-VL    | ⏳      | ⏳     | ⏳     | ⏳           |

### LLaVA 1.5 Complete Results (All Datasets)

| Dataset   | Seed 0  | Seed 1 | Seed 2 | **Mean(SD)** |
|-----------|---------|--------|--------|--------------|
| InfoSeek  | 57.23%  | ⏳     | ⏳     | ⏳           |
| ViQuAE    | 57.07%  | ⏳     | ⏳     | ⏳           |
| A-OKVQA   | 59.71%  | ⏳     | ⏳     | ⏳           |
| OK-VQA    | 60.66%  | ⏳     | ⏳     | ⏳           |
| E-VQA     | 35.97%  | ⏳     | ⏳     | ⏳           |

---

## 🗓️ TIMELINE

| Week | Phase | Model | Datasets | Jobs | Status |
|------|-------|-------|----------|------|--------|
| 1 (Current) | 1 | LLaVA 1.5 | All 5 | 10 | 🏃 RUNNING |
| 2 | 2 | InstructBLIP | InfoSeek, ViQuAE | 6 | ⏳ Pending |
| 3 | 3 | Shikra | InfoSeek, ViQuAE | 6 | ⏳ Pending |
| 4 | 4 | MiniGPT-4 | InfoSeek, ViQuAE | 6 | ⏳ Pending |
| 5 | 5 | LLaVA-NEXT | InfoSeek, ViQuAE | 6 | ⏳ Pending |
| 6 | 6 | Qwen2.5-VL | InfoSeek, ViQuAE | 6 | ⏳ Pending |

**Total**: ~6 weeks, 34 jobs, 36 experiments (including existing seed 0)

---

## 🛠️ AUTOMATION SCRIPTS

### Batch Script Template
Create `slurm_jobs/run_model_multiseed.sh`:

```bash
#!/bin/bash
MODEL=$1  # instructblip, shikra, minigpt4, etc.

echo "Submitting $MODEL multi-seed experiments..."

sbatch slurm_jobs/run_infoseek_${MODEL}_multiseed.slurm
sbatch slurm_jobs/run_viquae_${MODEL}_multiseed.slurm

echo "Submitted 6 jobs for $MODEL (3 seeds × 2 datasets)"
```

Usage:
```bash
bash slurm_jobs/run_model_multiseed.sh instructblip
bash slurm_jobs/run_model_multiseed.sh shikra
bash slurm_jobs/run_model_multiseed.sh minigpt4
```

---

## 📊 MONITORING

### Daily Checks
```bash
# Check all running jobs
squeue -u $USER

# Monitor specific phase
watch -n 30 'squeue -u $USER | grep tcvm'

# Check completion
ls -lh results/*/infoseek/*seed*.* results/*/viquae/*seed*.*
```

### Weekly Aggregation
```bash
# After each phase completes
python scripts/aggregate_tcvm_multiseed.py --model llava1.5 --all
python scripts/aggregate_tcvm_multiseed.py --model instructblip --dataset infoseek
python scripts/aggregate_tcvm_multiseed.py --model instructblip --dataset viquae
# etc...
```

---

## 🎯 SUCCESS CRITERIA

For each model-dataset combination:
- ✅ 3 seeds successfully completed
- ✅ Results aggregated with mean(SD)
- ✅ Standard deviation < 2% (indicates stability)
- ✅ Documentation updated

---

## 📁 RESULTS ORGANIZATION

Final structure:
```
results/
├── llava1.5/
│   ├── infoseek/
│   │   ├── infoseek_tcvm_seed0_results.jsonl
│   │   ├── infoseek_tcvm_seed1_results.jsonl
│   │   ├── infoseek_tcvm_seed2_results.jsonl
│   │   └── accuracy_multiseed_summary.txt
│   ├── viquae/ (same)
│   ├── aokvqa/ (same)
│   ├── okvqa/ (same)
│   └── evqa/ (same)
├── instructblip/
│   ├── infoseek/ (3 seeds + summary)
│   └── viquae/ (3 seeds + summary)
├── shikra/
│   ├── infoseek/
│   └── viquae/
├── minigpt4/
│   ├── infoseek/
│   └── viquae/
├── llava-next/
│   ├── infoseek/
│   └── viquae/
└── qwen2.5-vl/
    ├── infoseek/
    └── viquae/
```

---

## 🚨 RISK MITIGATION

### Potential Issues
1. **Jobs fail due to OOM**: Increase memory allocation
2. **Seed variance too high**: Investigate hyperparameter sensitivity
3. **Model not available**: Skip and document
4. **Script incompatibility**: Debug and fix before full run

### Backup Plan
- Keep all individual seed results
- Can re-run specific seeds if needed
- All scripts version controlled

---

## 📝 DOCUMENTATION TO UPDATE

After completion:
1. `docs/TCVM_KAR_RESULTS.md` - Add multi-seed results
2. `docs/MULTISEED_TCVM_KAR_STATUS.md` - Final status
3. `results/RESULTS_SUMMARY.md` - Complete table
4. `docs/MULTI_MODEL_PLAN.md` - Mark phases complete
5. Create final paper-ready table

---

## 🎓 FINAL DELIVERABLE

**Paper-Ready Table** (Example):

```markdown
| Model         | InfoSeek      | ViQuAE        |
|---------------|---------------|---------------|
| LLaVA 1.5     | 57.2 (0.3)    | 57.1 (0.4)    |
| InstructBLIP  | XX.X (X.X)    | XX.X (X.X)    |
| Shikra        | XX.X (X.X)    | XX.X (X.X)    |
| MiniGPT-4     | XX.X (X.X)    | XX.X (X.X)    |
| LLaVA-NEXT    | XX.X (X.X)    | XX.X (X.X)    |
| Qwen2.5-VL    | XX.X (X.X)    | XX.X (X.X)    |

Note: Results reported as Mean(SD) across 3 random seeds.
Config: TCVM-KAR with topk=20, α=1.0, β=0.7
```

---

**Document Version**: 1.0
**Created**: March 25, 2026, 21:15 AEDT
**Status**: Phase 1 (LLaVA 1.5) actively running
**Next Action**: Wait for Phase 1 completion (~48 hours)
