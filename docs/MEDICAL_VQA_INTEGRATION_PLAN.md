# Medical VQA Integration Plan

**Date**: 2026-04-16
**Purpose**: Integrate medical VQA datasets with medical Wikipedia knowledge base for ALFAR

## Overview

We are integrating medical-domain VQA datasets with a medical Wikipedia knowledge base to evaluate ALFAR's performance on specialized medical visual question answering tasks.

## Available Medical VQA Datasets

### 1. VQA-RAD (Radiology VQA)

**Statistics**:
- Questions: 2,248 question-answer pairs
- Images: 315 radiology images (X-rays, CT scans, MRI)
- Question Types: Open-ended and Yes/No questions
- Domain: Radiology, medical imaging

**Source**:
- Hugging Face: `flaviagiammarino/vqa-rad`
- Original: Nature Scientific Data (2018)
- License: CC0 1.0 Universal (Public Domain)

**Example Questions**:
- "What organ is shown in this image?"
- "Is there evidence of pneumonia?"
- "What is the abnormality in this X-ray?"

**Relevance to Medical Wikipedia**:
- Anatomy articles (organs, body parts)
- Disease articles (pneumonia, fractures, etc.)
- Medical imaging terminology

### 2. PathVQA (Pathology VQA)

**Statistics**:
- Questions: 32,799 open-ended questions
- Images: 4,998 pathology images (histopathology, microscopy)
- Question Types: Open-ended
- Domain: Pathology, cellular biology, diseases

**Source**:
- Hugging Face: `flaviagiammarino/path-vqa`
- GitHub: https://github.com/UCSD-AI4H/PathVQA
- Paper: ACL 2021 (https://arxiv.org/abs/2003.10286)
- License: MIT

**Example Questions**:
- "What type of cancer is shown?"
- "What is the staining method used?"
- "What cellular structure is visible?"

**Relevance to Medical Wikipedia**:
- Pathology articles (cancer types, diseases)
- Cellular biology articles
- Medical procedures (staining methods)
- Disease diagnosis

## Medical Wikipedia Integration

### Current Status

**Medical Wikipedia Download**:
- ✅ Downloading 1,755 medical articles
- ✅ Format matches existing ALFAR wiki format
- ⏳ In progress: ~17 minutes total

**Categories Covered** (14 categories):
1. Medical equipment (331 articles)
2. Medical signs (320 articles)
3. Medical treatments (249 articles)
4. Medical tests (205 articles)
5. Anatomy (135 articles)
6. Surgery (124 articles)
7. Pathology (106 articles)
8. Pharmacology (75 articles)
9. Medical specialties (68 articles)
10. Drugs (57 articles)
11. Medical procedures (51 articles)
12. Medicine (43 articles)
13. Diseases and disorders (31 articles)
14. Diagnostic medicine (0 articles)

### Integration Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Medical VQA Pipeline                      │
└─────────────────────────────────────────────────────────────┘

1. Data Preparation
   ├── Medical Wikipedia (1,755 articles) ──────┐
   ├── VQA-RAD Dataset (2,248 QA pairs)         │
   └── PathVQA Dataset (32,799 QA pairs)        │
                                                 │
2. Knowledge Base Setup                          │
   ├── medical_wiki_with_image.npy    <─────────┘
   ├── medical_wiki_map.npy
   └── Preprocessing & indexing

3. Retrieval System
   ├── Generate embeddings for medical wiki
   ├── Index medical knowledge
   └── Create retrieval indices
       ├── vqarad_retrieval_indices_50.npy
       └── pathvqa_retrieval_indices_50.npy

4. ALFAR Experiments
   ├── No-Context Baseline
   ├── Regular MRAG (context only)
   └── ALFAR (context + amplification)

5. Evaluation
   ├── VQA-RAD metrics
   ├── PathVQA metrics
   └── Analysis & comparison
```

## Implementation Steps

### Step 1: Download Medical VQA Datasets ✅

```bash
# Download VQA-RAD and PathVQA
python scripts/download_medical_vqa_datasets.py \
    --output-dir data/eval_data/medical \
    --datasets all
```

**Output Structure**:
```
data/eval_data/medical/
├── vqarad/
│   ├── train.json
│   ├── test.json
│   ├── images/
│   └── dataset_info.json
└── pathvqa/
    ├── train.json
    ├── test.json
    ├── images/
    └── dataset_info.json
```

### Step 2: Complete Medical Wikipedia Download ⏳

```bash
# Currently running (17 min ETA)
# Output: data/wiki/medical_wiki_with_image.npy
```

### Step 3: Create Wiki Map Files

```bash
python scripts/create_medical_wiki_map.py \
    --wiki-file data/wiki/medical_wiki_with_image.npy \
    --output data/wiki/medical_wiki_map.npy
```

### Step 4: Generate Retrieval Indices

Create a script to generate retrieval indices for medical datasets:

```bash
python scripts/generate_medical_retrieval_indices.py \
    --wiki data/wiki/medical_wiki_with_image.npy \
    --dataset vqarad \
    --output data/retrieval_result/medical/vqarad_indices_50.npy

python scripts/generate_medical_retrieval_indices.py \
    --wiki data/wiki/medical_wiki_with_image.npy \
    --dataset pathvqa \
    --output data/retrieval_result/medical/pathvqa_indices_50.npy
```

### Step 5: Adapt ALFAR Scripts for Medical VQA

Create medical VQA evaluation scripts:

```bash
experiments/eval/
├── alfar_medical_vqarad.py      # ALFAR for VQA-RAD
├── alfar_medical_pathvqa.py     # ALFAR for PathVQA
├── no_context_medical.py        # No-context baseline
└── regular_mrag_medical.py      # Regular MRAG baseline
```

### Step 6: Run Experiments

```bash
# VQA-RAD experiments
sbatch slurm_jobs/run_medical_vqarad_alfar.slurm
sbatch slurm_jobs/run_medical_vqarad_nocontext.slurm
sbatch slurm_jobs/run_medical_vqarad_regularmrag.slurm

# PathVQA experiments
sbatch slurm_jobs/run_medical_pathvqa_alfar.slurm
sbatch slurm_jobs/run_medical_pathvqa_nocontext.slurm
sbatch slurm_jobs/run_medical_pathvqa_regularmrag.slurm
```

### Step 7: Evaluate Results

```bash
# VQA-RAD evaluation
python evaluation/eval_medical_vqa.py \
    --dataset vqarad \
    --preds results/medical/vqarad_alfar_results.json

# PathVQA evaluation
python evaluation/eval_medical_vqa.py \
    --dataset pathvqa \
    --preds results/medical/pathvqa_alfar_results.json
```

## Expected Benefits

### Why Medical Wikipedia + Medical VQA?

**Domain Alignment**:
- Medical Wikipedia articles match medical VQA question topics
- Specialized terminology coverage
- Deeper domain knowledge than general Wikipedia

**Retrieval Quality**:
- Better context relevance for medical questions
- Medical-specific concepts (diseases, anatomy, procedures)
- Reduced noise from non-medical articles

**ALFAR Performance**:
- Test ALFAR on specialized domain
- Evaluate context amplification with domain-specific knowledge
- Compare general vs. medical knowledge bases

### Performance Expectations

| Method | General VQA | Medical VQA (Expected) |
|--------|-------------|------------------------|
| No-Context | Baseline | Lower (limited medical knowledge) |
| Regular MRAG (General Wiki) | ~Equal to No-Context | Noisy context |
| Regular MRAG (Medical Wiki) | N/A | Better context quality |
| ALFAR (General Wiki) | +14% | Unknown |
| **ALFAR (Medical Wiki)** | N/A | **Best (domain + amplification)** |

## Research Questions

1. **Does medical Wikipedia improve retrieval quality?**
   - Compare retrieval relevance: general vs. medical wiki

2. **Does ALFAR benefit from domain-specific knowledge?**
   - Compare ALFAR performance: general vs. medical wiki

3. **Are medical VQA questions more knowledge-intensive?**
   - Analyze question types and knowledge requirements

4. **How does ALFAR handle specialized medical terminology?**
   - Error analysis on medical terms

## Evaluation Metrics

### VQA-RAD Metrics
- Accuracy (exact match)
- Open-ended accuracy
- Yes/No accuracy
- Question-type breakdown

### PathVQA Metrics
- Accuracy (exact match)
- BLEU score (for open-ended)
- Domain-specific metrics (pathology terms)

### Analysis Metrics
- Retrieval quality (context relevance)
- Context utilization (attention weights)
- Error categories (medical vs. visual)

## Timeline

| Task | Status | ETA |
|------|--------|-----|
| Medical Wikipedia download | ⏳ In Progress | ~15 min |
| Medical VQA datasets download | ⏳ Pending | 30 min |
| Wiki map creation | ⏳ Pending | 5 min |
| Retrieval indices generation | ⏳ Pending | 2 hours |
| Script adaptation | ⏳ Pending | 1 day |
| Experiments (3 methods × 2 datasets) | ⏳ Pending | 6-12 hours |
| Evaluation & analysis | ⏳ Pending | 1 day |

**Total Estimated Time**: 2-3 days

## File Structure

```
ALFAR/
├── data/
│   ├── wiki/
│   │   ├── medical_wiki_with_image.npy       # Medical Wikipedia
│   │   └── medical_wiki_map.npy              # Index mapping
│   │
│   ├── eval_data/medical/
│   │   ├── vqarad/                           # VQA-RAD dataset
│   │   └── pathvqa/                          # PathVQA dataset
│   │
│   └── retrieval_result/medical/
│       ├── vqarad_indices_50.npy             # Retrieval indices
│       └── pathvqa_indices_50.npy
│
├── experiments/eval/
│   ├── alfar_medical_vqarad.py               # Medical VQA scripts
│   ├── alfar_medical_pathvqa.py
│   ├── no_context_medical.py
│   └── regular_mrag_medical.py
│
├── evaluation/
│   └── eval_medical_vqa.py                   # Evaluation script
│
├── results/medical/
│   ├── vqarad_alfar_results.json
│   ├── vqarad_nocontext_results.json
│   ├── vqarad_regularmrag_results.json
│   ├── pathvqa_alfar_results.json
│   ├── pathvqa_nocontext_results.json
│   └── pathvqa_regularmrag_results.json
│
└── docs/
    ├── MEDICAL_VQA_INTEGRATION_PLAN.md       # This file
    ├── MEDICAL_WIKI_DOWNLOAD_STATUS.md
    └── MEDICAL_VQA_RESULTS.md                # Results documentation
```

## Resources

### Datasets
- VQA-RAD: https://huggingface.co/datasets/flaviagiammarino/vqa-rad
- PathVQA: https://huggingface.co/datasets/flaviagiammarino/path-vqa
- Medical Wikipedia: Custom download from WikiProject Medicine

### Papers
- VQA-RAD: "A dataset of clinically generated visual questions and answers about radiology images" (Nature 2018)
- PathVQA: "PathVQA: 30000+ Questions for Medical Visual Question Answering" (ACL 2021)
- ALFAR: "Boosting Knowledge Utilization in MLLMs" (OpenReview)

### Tools
- Hugging Face Datasets: For downloading medical VQA datasets
- Wikipedia API: For downloading medical Wikipedia
- LLaVA-Med: Potential medical VLM baseline

## Next Actions

1. ✅ Download medical Wikipedia (in progress)
2. ⏳ Download VQA-RAD and PathVQA datasets
3. ⏳ Create wiki map files
4. ⏳ Generate retrieval indices
5. ⏳ Adapt ALFAR scripts for medical VQA
6. ⏳ Run baseline experiments
7. ⏳ Evaluate and analyze results
8. ⏳ Document findings

---

**Last Updated**: 2026-04-16
**Maintained by**: ALFAR Project Team
