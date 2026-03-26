# TCVM-KAR Multi-Model Results Summary

**Last Updated**: March 25, 2026
**Method**: TCVM-KAR (Token Contrastive Visual Masking with Knowledge-Aware Router)

## Results Organization Structure

```
results/
├── llava1.5/              # LLaVA 1.5-7B results
│   ├── infoseek/
│   │   ├── infoseek_tcvm_results.jsonl
│   │   └── accuracy.txt
│   ├── viquae/
│   │   ├── viquae_tcvm_results.jsonl
│   │   └── accuracy.txt
│   ├── aokvqa/
│   │   ├── aokvqa_tcvm_results.csv
│   │   └── accuracy.txt
│   ├── okvqa/
│   │   ├── okvqa_tcvm_results.csv
│   │   └── accuracy.txt
│   └── evqa/
│       ├── evqa_tcvm_results.json
│       └── accuracy.txt
├── instructblip/          # InstructBLIP results (TO BE ADDED)
│   ├── infoseek/
│   └── viquae/
├── shikra/                # Shikra results (TO BE ADDED)
│   ├── infoseek/
│   └── viquae/
├── minigpt4/              # MiniGPT-4 results (TO BE ADDED)
│   ├── infoseek/
│   └── viquae/
├── llava-next/            # LLaVA-NEXT results (TO BE ADDED)
│   ├── infoseek/
│   └── viquae/
└── qwen2.5-vl/            # Qwen2.5-VL results (TO BE ADDED)
    ├── infoseek/
    └── viquae/
```

---

## Current Results

### LLaVA 1.5-7B (COMPLETED ✅)

| Dataset   | Accuracy | Correct/Total | Config |
|-----------|----------|---------------|--------|
| InfoSeek  | **57.23%** | 1,717/3,000 | topk=20, α=1.0, β=0.7 |
| ViQuAE    | **57.07%** | 1,716/3,007 | topk=20, α=1.0, β=0.7 |
| A-OKVQA   | **59.71%** | 684/1,145   | topk=20, α=1.0, β=0.7 |
| OK-VQA    | **60.66%** | 3,061/5,046 | topk=20, α=1.0, β=0.7 |
| E-VQA     | **35.97%** | 50/139      | topk=20, α=1.0, β=0.7 |

**Status**: All experiments completed on March 25, 2026
**Location**: `results/llava1.5/`

---

## Planned Experiments

### Priority 1: InfoSeek & ViQuAE (Focus Datasets)

#### InstructBLIP (Pending)
- **Model**: BLIP-2 Vicuna Instruct 7B
- **Status**: ⏳ Script needs TCVM-KAR integration
- **Datasets**: InfoSeek, ViQuAE
- **Script**: `experiments/eval/alfar_mc_instructblip.py` (needs modification)

#### Shikra (Pending)
- **Model**: Shikra 7B
- **Status**: ⏳ Script needs TCVM-KAR integration
- **Datasets**: InfoSeek, ViQuAE
- **Script**: `experiments/eval/alfar_mc_shikra.py` (needs modification)

#### MiniGPT-4 (Pending)
- **Model**: MiniGPT-4 (Llama 2 Chat 7B)
- **Status**: ⏳ Script needs TCVM-KAR integration
- **Datasets**: InfoSeek, ViQuAE
- **Script**: `experiments/eval/alfar_mc_minigpt.py` (needs modification)

#### LLaVA-NEXT (Pending)
- **Model**: LLaVA-NEXT (LLaVA 1.6)
- **Status**: ⏳ Script needs to be created
- **Datasets**: InfoSeek, ViQuAE
- **Script**: To be created based on LLaVA structure

#### Qwen2.5-VL (Pending)
- **Model**: Qwen2.5-VL
- **Status**: ⏳ Script needs to be created
- **Datasets**: InfoSeek, ViQuAE
- **Script**: To be created

---

## Implementation Plan

### Step 1: Add TCVM-KAR Support to Existing Scripts

**Current ALFAR-only scripts**:
- `experiments/eval/alfar_mc_instructblip.py`
- `experiments/eval/alfar_mc_shikra.py`
- `experiments/eval/alfar_mc_minigpt.py`

**Required modifications** (based on LLaVA implementation):

1. **Import TCVM utilities** (already present via vcd_sample)
   ```python
   from vcd_sample import evolve_vcd_sampling
   evolve_vcd_sampling()  # Adds TCVM-KAR to generation
   ```

2. **Add TCVM arguments**:
   ```python
   parser.add_argument("--use_tcvm", action="store_true")
   parser.add_argument("--tcvm_topk", type=int, default=20)
   parser.add_argument("--tcvm_alpha", type=float, default=1.0)
   parser.add_argument("--tcvm_beta", type=float, default=0.7)
   parser.add_argument("--tcvm_mask_strategy", type=str, default='zero')
   ```

3. **Pass TCVM parameters to generate()**:
   ```python
   output = model.generate(
       # existing args...
       use_tcvm=args.use_tcvm,
       tcvm_topk=args.tcvm_topk,
       tcvm_alpha=args.tcvm_alpha,
       tcvm_beta=args.tcvm_beta,
       tcvm_mask_strategy=args.tcvm_mask_strategy,
   )
   ```

### Step 2: Create SLURM Scripts

For each model-dataset combination, create SLURM job scripts:

**Template**: `slurm_jobs/run_{dataset}_{model}_tcvm.slurm`

Example structure:
```bash
#!/bin/bash
#SBATCH --job-name=tcvm_{model}_{dataset}
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate

python experiments/eval/alfar_mc_{model}.py \
    --dataset {dataset} \
    --use_tcvm \
    --tcvm_topk 20 \
    --tcvm_alpha 1.0 \
    --tcvm_beta 0.7 \
    --answers-file results/{model}/{dataset}/{dataset}_tcvm_results.jsonl
```

### Step 3: Execution Order

**Phase 1: InstructBLIP** (Priority)
1. Modify `alfar_mc_instructblip.py` to add TCVM support
2. Test on small subset
3. Run InfoSeek + ViQuAE experiments
4. Evaluate and save results

**Phase 2: Shikra**
1. Modify `alfar_mc_shikra.py` to add TCVM support
2. Run InfoSeek + ViQuAE experiments
3. Evaluate and save results

**Phase 3: MiniGPT-4**
1. Modify `alfar_mc_minigpt.py` to add TCVM support
2. Run InfoSeek + ViQuAE experiments
3. Evaluate and save results

**Phase 4: LLaVA-NEXT** (if available)
1. Create new script based on LLaVA structure
2. Run InfoSeek + ViQuAE experiments

**Phase 5: Qwen2.5-VL** (if available)
1. Create new script or adapt existing
2. Run InfoSeek + ViQuAE experiments

---

## Expected Timeline

| Phase | Model | Datasets | Est. Time |
|-------|-------|----------|-----------|
| 1 | InstructBLIP | InfoSeek, ViQuAE | ~48 hours |
| 2 | Shikra | InfoSeek, ViQuAE | ~48 hours |
| 3 | MiniGPT-4 | InfoSeek, ViQuAE | ~48 hours |
| 4 | LLaVA-NEXT | InfoSeek, ViQuAE | ~48 hours |
| 5 | Qwen2.5-VL | InfoSeek, ViQuAE | ~48 hours |
| **Total** | **5 models** | **10 experiments** | **~10 days** |

---

## Configuration

All TCVM-KAR experiments use identical configuration for fair comparison:

```python
use_tcvm = True
tcvm_topk = 20              # Top-K tokens to mask
tcvm_alpha = 1.0            # Contrastive penalty weight
tcvm_beta = 0.7             # Plausibility threshold (APC)
tcvm_mask_strategy = 'zero' # Zero-out masked tokens
seed = 0                    # Random seed
```

---

## Evaluation

Results will be evaluated using:
- **InfoSeek**: `python evaluation/eval_mc.py --dataset infoseek`
- **ViQuAE**: `python evaluation/eval_mc.py --dataset viquae`

Each result folder will contain:
- Raw predictions file (`.jsonl` format)
- `accuracy.txt` with summary metrics

---

## Notes

### Current Status (March 25, 2026)
- ✅ LLaVA 1.5: All 5 datasets complete
- ⏳ InstructBLIP: Ready to implement TCVM support
- ⏳ Shikra: Ready to implement TCVM support
- ⏳ MiniGPT-4: Ready to implement TCVM support
- ⏳ LLaVA-NEXT: Needs new script
- ⏳ Qwen2.5-VL: Needs new script or adapter

### Model Availability
Check which models are available locally:
```bash
ls -la models/
```

### Priority Focus
Per user request, focus on **InfoSeek and ViQuAE** for the additional models before expanding to other datasets.

---

## References

- TCVM-KAR Implementation: `experiments/eval/vcd_sample.py`
- TCVM Utilities: `experiments/eval/tcvm_utils.py`
- LLaVA Example: `experiments/eval/alfar_mc_llava.py`
- Evaluation Scripts: `evaluation/eval_mc.py`

---

**Maintained by**: ALFAR Project Team
**Version**: 1.0
**Last Updated**: March 25, 2026
