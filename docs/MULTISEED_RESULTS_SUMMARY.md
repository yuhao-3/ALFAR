# Multiseed Experiment Results Summary

**Date**: March 23, 2026
**Seeds**: 0, 1, 2, 3, 4 (5 total)
**Methods**: ALFAR vs TCVM

## Overall Summary

| Dataset  | ALFAR (Mean ± Std)     | TCVM (Mean ± Std)      | Difference | Winner |
|----------|------------------------|------------------------|------------|--------|
| A-OKVQA  | 60.13% ± 0.31%        | 59.66% ± 0.32%        | +0.46%     | ✓ ALFAR |
| OKVQA    | 61.15% ± 0.15%        | 60.45% ± 0.23%        | +0.70%     | ✓ ALFAR |
| InfoSeek | 57.43% ± 0.26%        | 57.45% ± 0.32%        | -0.02%     | ✗ TCVM  |
| ViQuAE   | 56.30% ± 0.32%        | 56.74% ± 0.32%        | -0.44%     | ✗ TCVM  |

**Note**: EVQA results pending (requires TensorFlow installation)

---

## Detailed Results by Dataset

### A-OKVQA (Answerable OKVQA)

**ALFAR**: 60.13% ± 0.31%
- Seed 0: 60.23%
- Seed 1: 60.06%
- Seed 2: 60.49%
- Seed 3: 59.65%
- Seed 4: 60.20%

**TCVM**: 59.66% ± 0.32%
- Seed 0: 59.71%
- Seed 1: 60.12%
- Seed 2: 59.48%
- Seed 3: 59.27%
- Seed 4: 59.74%

**Difference**: +0.46% (ALFAR better)

---

### OKVQA (Outside Knowledge VQA)

**ALFAR**: 61.15% ± 0.15%
- Seed 0: 60.93%
- Seed 1: 61.26%
- Seed 2: 61.05%
- Seed 3: 61.28%
- Seed 4: 61.21%

**TCVM**: 60.45% ± 0.23%
- Seed 0: 60.66%
- Seed 1: 60.08%
- Seed 2: 60.40%
- Seed 3: 60.46%
- Seed 4: 60.64%

**Difference**: +0.70% (ALFAR better)

---

### InfoSeek

**ALFAR**: 57.43% ± 0.26%
- Seed 0: 57.30%
- Seed 1: 57.13%
- Seed 2: 57.37%
- Seed 3: 57.50%
- Seed 4: 57.83%

**TCVM**: 57.45% ± 0.32%
- Seed 0: 57.47%
- Seed 1: 57.20%
- Seed 2: 57.43%
- Seed 3: 57.17%
- Seed 4: 57.97%

**Difference**: -0.02% (TCVM slightly better, essentially tied)

---

### ViQuAE (Vietnamese Question Answering)

**ALFAR**: 56.30% ± 0.32%
- Seed 0: 56.07%
- Seed 1: 56.50%
- Seed 2: 56.27%
- Seed 3: 56.73%
- Seed 4: 55.94%

**TCVM**: 56.74% ± 0.32%
- Seed 0: 56.63%
- Seed 1: 57.13%
- Seed 2: 56.43%
- Seed 3: 57.03%
- Seed 4: 56.47%

**Difference**: -0.44% (TCVM better)

---

## Key Findings

1. **ALFAR performs better on knowledge-intensive QA tasks**:
   - A-OKVQA: +0.46% improvement
   - OKVQA: +0.70% improvement (largest gap)

2. **TCVM performs better on visual reasoning tasks**:
   - ViQuAE: +0.44% improvement
   - InfoSeek: +0.02% (essentially tied)

3. **Stability**: Both methods show good consistency across seeds with standard deviations typically 0.15-0.32%

4. **Overall**: Mixed results - method performance depends on task type

---

## LaTeX Table (for papers)

```latex
\begin{table}[h]
\centering
\begin{tabular}{l|c|c|c}
\hline
Dataset & ALFAR & TCVM & Diff \\
\hline
AOKVQA & $60.13 \pm 0.31$ & $59.66 \pm 0.32$ & $+0.46$ \\
OKVQA & $61.15 \pm 0.15$ & $60.45 \pm 0.23$ & $+0.70$ \\
INFOSEEK & $57.43 \pm 0.26$ & $57.45 \pm 0.32$ & $-0.02$ \\
VIQUAE & $56.30 \pm 0.32$ & $56.74 \pm 0.32$ & $-0.44$ \\
\hline
\end{tabular}
\caption{Comparison of ALFAR and TCVM across datasets (mean $\pm$ std)}
\end{table}
```

---

## Files Generated

1. **logs/multiseed_comprehensive_summary.txt**: Full detailed output
2. **logs/multiseed_results_summary.csv**: CSV format for analysis
3. **MULTISEED_RESULTS_SUMMARY.md**: This summary document
4. **Individual metrics files**: logs/*_seed*_metrics.txt

---

## Usage

To regenerate these results:

```bash
# Generate missing metrics
bash scripts/generate_missing_metrics.sh

# Calculate all averages
python scripts/calculate_all_averages.py

# Or use the aggregate script for individual dataset/method:
python scripts/aggregate_multiseed_results.py --dataset okvqa --method alfar
```
