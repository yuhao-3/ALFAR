#!/usr/bin/env python3
"""
Create visual diagram of bucket classification criteria.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_bucket_definition_diagram():
    """Create a clear visual diagram of bucket classification"""

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    title_text = "Bucket Classification Algorithm\nModel: LLaVA-1.5-7B (CLIP ViT-L/14 + Vicuna-7B v1.5)"
    ax.text(8, 9.5, title_text, ha='center', va='top', fontsize=16,
            fontweight='bold', family='monospace')

    # Input box
    input_box = FancyBboxPatch((0.5, 7.5), 4, 1.2, boxstyle="round,pad=0.1",
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, 8.5, 'INPUT', ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(2.5, 8.15, '• Question + Image', ha='center', va='top', fontsize=9)
    ax.text(2.5, 7.9, '• Gold Answers (10 annotations)', ha='center', va='top', fontsize=9)
    ax.text(2.5, 7.65, '• Retrieved Context (Wikipedia)', ha='center', va='top', fontsize=9)

    # Decision 1: No-context correct?
    y_decision1 = 6.3
    decision1_box = FancyBboxPatch((0.5, y_decision1-0.6), 4, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#FFE5B4', edgecolor='black', linewidth=2)
    ax.add_patch(decision1_box)
    ax.text(2.5, y_decision1, 'DECISION 1', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(2.5, y_decision1-0.3, 'No-Context Correct?', ha='center', va='center',
            fontsize=9, style='italic')

    # Arrow from input to decision 1
    arrow1 = FancyArrowPatch((2.5, 7.5), (2.5, y_decision1+0.4),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)

    # Decision 2: Context has answer?
    y_decision2 = 4.5
    decision2_box = FancyBboxPatch((0.5, y_decision2-0.6), 4, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#FFE5B4', edgecolor='black', linewidth=2)
    ax.add_patch(decision2_box)
    ax.text(2.5, y_decision2, 'DECISION 2', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(2.5, y_decision2-0.3, 'Context Contains Answer?', ha='center', va='center',
            fontsize=9, style='italic')

    # Arrow from decision 1 to decision 2
    arrow2 = FancyArrowPatch((2.5, y_decision1-0.6), (2.5, y_decision2+0.4),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)

    # Buckets - positioned based on decision outcomes
    bucket_y = 2
    bucket_width = 3.2
    bucket_height = 1.5

    # Corrective bucket (No-Ctx Wrong + Context Has Answer)
    corrective_x = 0.5
    corrective_box = FancyBboxPatch((corrective_x, bucket_y), bucket_width, bucket_height,
                                    boxstyle="round,pad=0.15",
                                    facecolor='#90EE90', edgecolor='darkgreen', linewidth=3)
    ax.add_patch(corrective_box)
    ax.text(corrective_x + bucket_width/2, bucket_y + 1.2, 'CORRECTIVE',
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkgreen')
    ax.text(corrective_x + bucket_width/2, bucket_y + 0.85, 'n=135 (11.8%)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(corrective_x + bucket_width/2, bucket_y + 0.55, '✗ No-Ctx Wrong',
            ha='center', va='center', fontsize=8)
    ax.text(corrective_x + bucket_width/2, bucket_y + 0.3, '✓ Context Has Answer',
            ha='center', va='center', fontsize=8)
    ax.text(corrective_x + bucket_width/2, bucket_y + 0.05, 'ALFAR: 72.59%',
            ha='center', va='center', fontsize=9, fontweight='bold', color='darkgreen')

    # Neutral bucket (No-Ctx Correct + Context Has Answer)
    neutral_x = 4.2
    neutral_box = FancyBboxPatch((neutral_x, bucket_y), bucket_width, bucket_height,
                                 boxstyle="round,pad=0.15",
                                 facecolor='#87CEEB', edgecolor='darkblue', linewidth=3)
    ax.add_patch(neutral_box)
    ax.text(neutral_x + bucket_width/2, bucket_y + 1.2, 'NEUTRAL',
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkblue')
    ax.text(neutral_x + bucket_width/2, bucket_y + 0.85, 'n=402 (35.1%)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(neutral_x + bucket_width/2, bucket_y + 0.55, '✓ No-Ctx Correct',
            ha='center', va='center', fontsize=8)
    ax.text(neutral_x + bucket_width/2, bucket_y + 0.3, '✓ Context Has Answer',
            ha='center', va='center', fontsize=8)
    ax.text(neutral_x + bucket_width/2, bucket_y + 0.05, 'ALFAR: 93.28%',
            ha='center', va='center', fontsize=9, fontweight='bold', color='darkblue')

    # Misleading bucket (No-Ctx Wrong + Context Lacks Answer)
    misleading_x = 7.9
    misleading_box = FancyBboxPatch((misleading_x, bucket_y), bucket_width, bucket_height,
                                    boxstyle="round,pad=0.15",
                                    facecolor='#FFB6C6', edgecolor='darkred', linewidth=3)
    ax.add_patch(misleading_box)
    ax.text(misleading_x + bucket_width/2, bucket_y + 1.2, 'MISLEADING',
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkred')
    ax.text(misleading_x + bucket_width/2, bucket_y + 0.85, 'n=292 (25.5%)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(misleading_x + bucket_width/2, bucket_y + 0.55, '✗ No-Ctx Wrong',
            ha='center', va='center', fontsize=8)
    ax.text(misleading_x + bucket_width/2, bucket_y + 0.3, '✗ Context Lacks Answer',
            ha='center', va='center', fontsize=8)
    ax.text(misleading_x + bucket_width/2, bucket_y + 0.05, 'ALFAR: 38.36%',
            ha='center', va='center', fontsize=9, fontweight='bold', color='darkred')

    # Other bucket (No-Ctx Correct + Context Lacks Answer) - grayed out
    other_x = 11.6
    other_box = FancyBboxPatch((other_x, bucket_y), bucket_width, bucket_height,
                               boxstyle="round,pad=0.15",
                               facecolor='#D3D3D3', edgecolor='gray', linewidth=2, linestyle='--')
    ax.add_patch(other_box)
    ax.text(other_x + bucket_width/2, bucket_y + 1.2, 'OTHER',
            ha='center', va='center', fontsize=11, fontweight='bold', color='gray')
    ax.text(other_x + bucket_width/2, bucket_y + 0.85, 'n=316 (27.6%)',
            ha='center', va='center', fontsize=9, fontweight='bold', color='gray')
    ax.text(other_x + bucket_width/2, bucket_y + 0.55, '✓ No-Ctx Correct',
            ha='center', va='center', fontsize=8, color='gray')
    ax.text(other_x + bucket_width/2, bucket_y + 0.3, '✗ Context Lacks Answer',
            ha='center', va='center', fontsize=8, color='gray')
    ax.text(other_x + bucket_width/2, bucket_y + 0.05, 'Excluded',
            ha='center', va='center', fontsize=9, fontweight='bold', style='italic', color='gray')

    # Arrows from decision to buckets
    # From decision 2 to Corrective
    arrow_corr = FancyArrowPatch((2.5, y_decision2-0.6), (corrective_x + bucket_width/2, bucket_y + bucket_height),
                                arrowstyle='->', mutation_scale=15, linewidth=1.5, color='darkgreen')
    ax.add_patch(arrow_corr)
    ax.text(1.5, 3.3, '✗ + ✓', ha='center', fontsize=8, color='darkgreen', fontweight='bold')

    # From decision 2 to Neutral
    arrow_neutral = FancyArrowPatch((2.5, y_decision2-0.6), (neutral_x + bucket_width/2, bucket_y + bucket_height),
                                   arrowstyle='->', mutation_scale=15, linewidth=1.5, color='darkblue')
    ax.add_patch(arrow_neutral)
    ax.text(3.5, 3.3, '✓ + ✓', ha='center', fontsize=8, color='darkblue', fontweight='bold')

    # From decision 2 to Misleading
    arrow_mislead = FancyArrowPatch((2.5, y_decision2-0.6), (misleading_x + bucket_width/2, bucket_y + bucket_height),
                                   arrowstyle='->', mutation_scale=15, linewidth=1.5, color='darkred')
    ax.add_patch(arrow_mislead)
    ax.text(5.5, 3.3, '✗ + ✗', ha='center', fontsize=8, color='darkred', fontweight='bold')

    # From decision 2 to Other
    arrow_other = FancyArrowPatch((2.5, y_decision2-0.6), (other_x + bucket_width/2, bucket_y + bucket_height),
                                 arrowstyle='->', mutation_scale=15, linewidth=1.5, color='gray', linestyle='--')
    ax.add_patch(arrow_other)
    ax.text(7.5, 3.3, '✓ + ✗', ha='center', fontsize=8, color='gray', fontweight='bold')

    # Key finding box
    key_box = FancyBboxPatch((0.5, 0.2), 14.5, 1.2, boxstyle="round,pad=0.1",
                             facecolor='#FFFF99', edgecolor='black', linewidth=3)
    ax.add_patch(key_box)
    ax.text(7.75, 1.15, 'KEY FINDING: 34% Performance Gap',
            ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(7.75, 0.8, 'Corrective (72.59%) - Misleading (38.36%) = 34.23% degradation',
            ha='center', va='center', fontsize=11)
    ax.text(7.75, 0.45, '→ Proves ALFAR blindly amplifies context without quality assessment',
            ha='center', va='center', fontsize=10, style='italic')

    # Legend for evaluation methods
    legend_x = 11
    legend_y = 6.5
    ax.text(legend_x, legend_y+0.3, 'Evaluation Methods:', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    ax.text(legend_x, legend_y-0.1, 'No-Ctx Correct: Fuzzy substring match', fontsize=7)
    ax.text(legend_x, legend_y-0.4, 'Context Has Answer: Gold answer in context', fontsize=7)
    ax.text(legend_x, legend_y-0.7, 'Dataset: A-OKVQA (1145 samples)', fontsize=7)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = create_bucket_definition_diagram()

    # Save
    output_dir = "results/bucket_analysis"
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(f"{output_dir}/bucket_classification_diagram.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_dir}/bucket_classification_diagram.png", dpi=300, bbox_inches='tight')

    print(f"Bucket classification diagram saved to:")
    print(f"  - {output_dir}/bucket_classification_diagram.pdf")
    print(f"  - {output_dir}/bucket_classification_diagram.png")

    plt.close()
