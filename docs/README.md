# ALFAR Documentation

This directory contains comprehensive documentation for the ALFAR (Adaptive Logit Fusion and Attention Re-weighting) framework.

## Documents

### [ALFAR Technical Documentation](ALFAR_TECHNICAL_DOCUMENTATION.md)
Complete technical documentation covering:
- **Adaptive Attention Allocation**: How the model dynamically re-weights attention between image and text
- **Adaptive Logit Fusion**: How contrastive decoding combines predictions from dual branches
- **Code Architecture**: Detailed explanation of all components
- **Evaluation Scripts**: How alfar_{dataset}_{model}.py scripts work
- **Parameter Guide**: Comprehensive explanation of all hyperparameters

## Quick Reference

### Key Mechanisms

#### Adaptive Attention Allocation
- **Location**: `experiments/eval/attention.py`
- **Formula**:
  ```
  Image: A_img' = A_img × (1 - ret_sim × α)
  Context: A_ctx' = A_ctx × (1 + α × weight)
  ```
- **Parameters**: `att_alpha`, `ret_sim`

#### Adaptive Logit Fusion
- **Location**: `experiments/eval/vcd_sample.py`
- **Formula**:
  ```
  cd_alpha = image_attn / context_attn
  L_contrast = (1 + 1/cd_alpha) × L_ctx - (1 - cd_alpha) × L_no_ctx
  ```
- **Parameters**: `cd_beta`

### Dataset Configurations

| Dataset   | att_alpha | ret_sim | cd_beta | Rationale                      |
|-----------|-----------|---------|---------|--------------------------------|
| InfoSeek  | 0.2       | Variable| 0.7     | Balanced image-text retrieval  |
| ViQuAE    | 0.2       | Variable| 0.7     | Balanced image-text retrieval  |
| A-OKVQA   | 0.4       | 0.9     | 0.7     | Heavy external knowledge need  |
| OK-VQA    | 0.4       | 0.9     | 0.7     | Heavy external knowledge need  |
| E-VQA     | 0.1       | 1.0     | 0.7     | Gold evidence (minimal tuning) |

### File Reference

**Core Implementation**:
- `experiments/eval/alfar_mc_llava.py` - Multiple-choice VQA (InfoSeek, ViQuAE)
- `experiments/eval/alfar_okvqa_llava.py` - Open-ended VQA (A-OKVQA, OK-VQA)
- `experiments/eval/alfar_evqa_llava.py` - Evidence-based VQA (E-VQA)
- `experiments/eval/attention.py` - Adaptive attention mechanism
- `experiments/eval/vcd_sample.py` - Contrastive decoding sampling
- `experiments/llava/model/language_model/llava_llama.py` - Modified LLaVA model

**Evaluation**:
- `evaluation/eval_mc.py` - Multiple-choice evaluation
- `evaluation/eval_okvqa.py` - OK-VQA evaluation
- `evaluation/eval_evqa.py` - E-VQA evaluation

**SLURM Jobs**:
- `slurm_jobs/run_*_eval.slurm` - GPU job scripts for evaluation

**Data**:
- `data/eval_data/` - Question datasets
- `data/retrieval_result/` - Retrieved knowledge indices
- `data/wiki/` - Wikipedia knowledge base

## Getting Started

1. Read [ALFAR Technical Documentation](ALFAR_TECHNICAL_DOCUMENTATION.md) for complete understanding
2. Check dataset configurations for your use case
3. Run evaluation with appropriate SLURM scripts
4. Adjust `att_alpha` and `cd_beta` based on your context quality

## Questions?

For detailed explanations of:
- How attention reweighting works → See "Adaptive Attention Allocation" section
- How logit fusion works → See "Adaptive Logit Fusion" section
- How to run evaluation → See "Evaluation Scripts" section
- What parameters to use → See "Key Parameters" and "Dataset-Specific Configurations" sections
