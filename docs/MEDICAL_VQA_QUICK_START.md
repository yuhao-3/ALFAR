# Medical VQA with ALFAR - Quick Start Guide

**Goal**: Run ALFAR on Medical VQA datasets (VQA-RAD & PathVQA) with Medical Wikipedia RAG

**Date**: 2026-04-16
**Status**: Downloads in progress

---

## Current Progress

### ✅ Completed
- [x] Created medical Wikipedia download script
- [x] Created medical VQA download script
- [x] Created wiki map generation script
- [x] Installed required dependencies (`datasets` library)
- [x] Started downloads (running in background)

### 🔄 In Progress
- [ ] Downloading Medical Wikipedia (~50% complete, ~10 min remaining)
  - **Status**: 410+/1,755 articles
  - **File**: `data/wiki/medical_wiki_with_image.npy`

- [ ] Downloading VQA-RAD & PathVQA datasets
  - **Status**: Starting...
  - **Files**: `data/eval_data/medical/vqarad/` & `pathvqa/`

### ⏳ Next Steps
1. Create medical wiki map
2. Generate retrieval indices
3. Adapt ALFAR scripts for medical VQA
4. Run experiments

---

## Quick Commands

### Check Download Progress
```bash
# Medical Wikipedia progress
tail -f logs/medical_wiki_download.log

# Medical VQA datasets progress
tail -f logs/medical_vqa_download.log
```

### After Downloads Complete

**1. Create Wiki Map**:
```bash
python scripts/create_medical_wiki_map.py \
    --wiki-file data/wiki/medical_wiki_with_image.npy \
    --output data/wiki/medical_wiki_map.npy
```

**2. Test Medical Wiki Format**:
```python
import numpy as np

# Load medical wiki
med_wiki = np.load('data/wiki/medical_wiki_with_image.npy', allow_pickle=True).item()

# Check structure
print(f"Total articles: {len(med_wiki)}")
sample_id = list(med_wiki.keys())[0]
print(f"Sample entry structure:")
print(f"  - wikipedia_summary: {len(med_wiki[sample_id]['wikipedia_summary'])} chars")
print(f"  - wikipedia_content: {len(med_wiki[sample_id]['wikipedia_content'])} chars")
```

**3. Test Medical VQA Data**:
```python
import json

# Load VQA-RAD
with open('data/eval_data/medical/vqarad/test.json') as f:
    vqarad = json.load(f)
print(f"VQA-RAD test samples: {len(vqarad)}")

# Load PathVQA
with open('data/eval_data/medical/pathvqa/test.json') as f:
    pathvqa = json.load(f)
print(f"PathVQA test samples: {len(pathvqa)}")
```

---

## Medical VQA Datasets Overview

### VQA-RAD (Radiology)
- **Type**: Open-ended & Yes/No questions
- **Images**: 315 radiology images (X-rays, CT, MRI)
- **Questions**: 2,248 QA pairs
- **Example**: "What organ is enlarged?" → "Heart"

**Relevant Medical Wiki Categories**:
- Anatomy (135 articles)
- Diagnostic medicine
- Medical signs (320 articles)
- Pathology (106 articles)

### PathVQA (Pathology)
- **Type**: Open-ended questions
- **Images**: 4,998 pathology microscopy images
- **Questions**: 32,799 QA pairs
- **Example**: "What type of cancer cells are shown?" → "Adenocarcinoma"

**Relevant Medical Wiki Categories**:
- Pathology (106 articles)
- Diseases and disorders (31 articles)
- Medical tests (205 articles)
- Anatomy (135 articles)

---

## ALFAR Pipeline for Medical VQA

### 1. Data Flow
```
Medical Wikipedia (1,755 articles)
         ↓
Medical Wiki Map (index mapping)
         ↓
Retrieval System (top-k medical articles)
         ↓
Medical VQA Question + Retrieved Context
         ↓
ALFAR (Attention Reallocation + Logits Fusion)
         ↓
Answer
```

### 2. Three-Way Comparison

| Method | Context in Prompt | ALFAR Intervention | Purpose |
|--------|------------------|-------------------|---------|
| **No-Context** | ✗ | ✗ | Baseline (parametric only) |
| **Regular MRAG** | ✓ (Medical Wiki) | ✗ | Test context value |
| **ALFAR** | ✓ (Medical Wiki) | ✓ | Test amplification |

### 3. Key Differences from General VQA

| Aspect | General VQA | Medical VQA |
|--------|-------------|-------------|
| **Images** | Natural scenes | X-rays, microscopy |
| **Questions** | General knowledge | Medical expertise |
| **Answers** | Common words | Medical terminology |
| **Knowledge Base** | General Wikipedia | Medical Wikipedia |
| **Context Relevance** | Variable | High (domain-specific) |

---

## Expected Outcomes

### Hypothesis
Medical Wikipedia should provide **higher quality context** for medical VQA compared to general Wikipedia:

1. **Better Retrieval**: Medical articles match medical queries better
2. **Less Noise**: No irrelevant non-medical articles
3. **Specialized Terms**: Medical terminology coverage
4. **ALFAR Benefit**: Context amplification should help more with quality context

### Metrics to Track

**VQA-RAD**:
- Overall accuracy
- Open-ended accuracy
- Yes/No accuracy
- Question-type breakdown

**PathVQA**:
- Overall accuracy
- Answer type distribution
- Medical term coverage

**Analysis**:
- Context relevance (manual inspection)
- Attention weights (ALFAR vs baseline)
- Error categories (visual vs. knowledge)

---

## File Structure (After Setup)

```
ALFAR/
├── data/
│   ├── wiki/
│   │   ├── medical_wiki_with_image.npy    # 1,755 medical articles
│   │   └── medical_wiki_map.npy           # Index mapping
│   │
│   ├── eval_data/medical/
│   │   ├── vqarad/
│   │   │   ├── train.json
│   │   │   ├── test.json
│   │   │   ├── images/                     # 315 images
│   │   │   └── dataset_info.json
│   │   └── pathvqa/
│   │       ├── train.json
│   │       ├── test.json
│   │       ├── images/                     # 4,998 images
│   │       └── dataset_info.json
│   │
│   └── retrieval_result/medical/
│       ├── vqarad_indices_50.npy
│       └── pathvqa_indices_50.npy
│
├── experiments/eval/
│   ├── alfar_medical_vqarad.py
│   ├── alfar_medical_pathvqa.py
│   ├── no_context_medical_vqarad.py
│   └── regular_mrag_medical_vqarad.py
│
├── results/medical/
│   ├── vqarad/
│   └── pathvqa/
│
└── logs/
    ├── medical_wiki_download.log
    └── medical_vqa_download.log
```

---

## Next Actions (Automated)

Once downloads complete, the pipeline will:

1. **Verify downloads**:
   - Check medical wiki format
   - Check VQA dataset structure

2. **Create wiki map**:
   - Generate index mapping
   - Verify compatibility

3. **Adapt ALFAR scripts**:
   - Copy from existing ALFAR scripts
   - Modify for medical VQA format
   - Add medical wiki path

4. **Ready to run**:
   - Scripts ready for SLURM
   - Can start experiments

---

## Estimated Timeline

| Task | Time | Status |
|------|------|--------|
| Medical Wikipedia download | ~17 min | 🔄 In progress |
| Medical VQA download | ~10 min | 🔄 Starting |
| Wiki map creation | ~1 min | ⏳ Pending |
| Retrieval indices | ~30 min | ⏳ Pending |
| Script adaptation | ~15 min | ⏳ Pending |
| **Total Setup Time** | **~1 hour** | |
| Experiments (3 methods × 2 datasets) | 6-12 hours | ⏳ Pending |

---

**Last Updated**: 2026-04-16 15:51 AEDT
