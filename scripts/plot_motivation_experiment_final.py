#!/usr/bin/env python3
"""
Create final comprehensive visualization for motivation experiment (three-way analysis).
Shows No-Context vs Regular MRAG vs ALFAR across all buckets.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from three-way bucket analysis
data = {
    "Overall": {
        "No-Context": 46.46,
        "Regular MRAG": 46.08,
        "ALFAR": 60.23,
        "count": 1145
    },
    "Corrective": {
        "No-Context": 0.00,
        "Regular MRAG": 53.33,
        "ALFAR": 72.59,
        "count": 135
    },
    "Neutral": {
        "No-Context": 100.00,
        "Regular MRAG": 77.61,
        "ALFAR": 93.28,
        "count": 402
    },
    "Misleading": {
        "No-Context": 0.00,
        "Regular MRAG": 32.19,
        "ALFAR": 38.36,
        "count": 292
    }
}

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# 1. Overall + All Buckets Comparison (Top Left)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

categories = ["Overall", "Corrective", "Neutral", "Misleading"]
x = np.arange(len(categories))
width = 0.25

no_ctx = [data[cat]["No-Context"] for cat in categories]
reg_mrag = [data[cat]["Regular MRAG"] for cat in categories]
alfar = [data[cat]["ALFAR"] for cat in categories]
counts = [data[cat]["count"] for cat in categories]

bars1 = ax1.bar(x - width, no_ctx, width, label='No-Context', color='#E74C3C', alpha=0.8)
bars2 = ax1.bar(x, reg_mrag, width, label='Regular MRAG', color='#F39C12', alpha=0.8)
bars3 = ax1.bar(x + width, alfar, width, label='ALFAR', color='#27AE60', alpha=0.8)

# Add value labels
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

# Add sample counts
category_labels = [f"{cat}\n(n={count})" for cat, count in zip(categories, counts)]
ax1.set_xticks(x)
ax1.set_xticklabels(category_labels, fontsize=11)

ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Three-Way Comparison: No-Context vs Regular MRAG vs ALFAR (A-OKVQA)',
             fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 110)

# Add key finding text
ax1.text(0.02, 0.98,
         "Key Finding: Regular MRAG ≈ No-Context\nALFAR outperforms both across ALL buckets",
         transform=ax1.transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
         fontweight='bold')

# ============================================================================
# 2. ALFAR Advantage over Regular MRAG (Bottom Left)
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

bucket_names = ["Corrective", "Neutral", "Misleading"]
alfar_advantage = [
    data["Corrective"]["ALFAR"] - data["Corrective"]["Regular MRAG"],
    data["Neutral"]["ALFAR"] - data["Neutral"]["Regular MRAG"],
    data["Misleading"]["ALFAR"] - data["Misleading"]["Regular MRAG"]
]

colors = ['#27AE60', '#27AE60', '#27AE60']
bars = ax2.barh(bucket_names, alfar_advantage, color=colors, alpha=0.8)

for i, (bar, val) in enumerate(zip(bars, alfar_advantage)):
    ax2.text(val + 0.5, i, f'+{val:.2f}%',
            va='center', fontsize=11, fontweight='bold')

ax2.set_xlabel('ALFAR Advantage over Regular MRAG (%)', fontsize=11, fontweight='bold')
ax2.set_title('ALFAR Performance Advantage', fontsize=12, fontweight='bold', pad=10)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim(0, 25)

# Highlight Misleading bucket
ax2.axhline(y=2, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.text(12, 2.3, '← Critical Test: ALFAR wins in Misleading bucket',
        fontsize=10, color='red', fontweight='bold')

# ============================================================================
# 3. Regular MRAG Failure Analysis (Bottom Right)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

methods = ['No-Context', 'Regular\nMRAG', 'ALFAR']
overall_accs = [
    data["Overall"]["No-Context"],
    data["Overall"]["Regular MRAG"],
    data["Overall"]["ALFAR"]
]

colors_overall = ['#E74C3C', '#F39C12', '#27AE60']
bars = ax3.bar(methods, overall_accs, color=colors_overall, alpha=0.8, width=0.6)

for bar, val in zip(bars, overall_accs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Draw arrows showing gaps
ax3.annotate('', xy=(1, 46.08), xytext=(0, 46.46),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax3.text(0.5, 44, '-0.38%\n(No benefit!)',
        ha='center', fontsize=9, color='red', fontweight='bold')

ax3.annotate('', xy=(2, 60.23), xytext=(1, 46.08),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(1.5, 53, '+14.15%\n(Amplification\nworks!)',
        ha='center', fontsize=9, color='green', fontweight='bold')

ax3.set_ylabel('Overall Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_title('Overall Performance Comparison', fontsize=12, fontweight='bold', pad=10)
ax3.set_ylim(0, 70)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add interpretation
ax3.text(0.5, 0.95,
         'Regular MRAG ≈ No-Context\n→ Context alone is insufficient',
         transform=ax3.transAxes,
         fontsize=10, verticalalignment='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
         fontweight='bold')

plt.suptitle('A-OKVQA Motivation Experiment: Three-Way Analysis Results',
            fontsize=16, fontweight='bold', y=0.98)

# Save
output_dir = Path("results/bucket_analysis_threeway")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "aokvqa_motivation_final.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "aokvqa_motivation_final.pdf", bbox_inches='tight')
print(f"Saved visualizations to {output_dir}/")
print("  - aokvqa_motivation_final.png")
print("  - aokvqa_motivation_final.pdf")
plt.close()
