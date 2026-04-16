#!/usr/bin/env python3
"""
Plot bucket analysis results for motivation experiment.

Creates a grouped bar chart showing performance across three context reliability buckets.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_bucket_results(stats_file, output_file, dataset_name):
    """Generate visualization of bucket analysis results"""

    # Load statistics
    with open(stats_file, 'r') as f:
        results = json.load(f)

    # Extract data for plotting
    buckets = ["Corrective", "Neutral", "Misleading"]
    bucket_labels = []
    no_ctx_accs = []
    alfar_accs = []
    counts = []

    for bucket in buckets:
        if bucket in results:
            bucket_labels.append(f"{bucket}\n(n={results[bucket]['count']})")
            no_ctx_accs.append(results[bucket]['no_context_acc'])
            alfar_accs.append(results[bucket]['alfar_acc'])
            counts.append(results[bucket]['count'])
        else:
            bucket_labels.append(f"{bucket}\n(n=0)")
            no_ctx_accs.append(0)
            alfar_accs.append(0)
            counts.append(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(bucket_labels))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(x - width/2, no_ctx_accs, width, label='No Context (Parametric)',
                   color='#808080', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, alfar_accs, width, label='ALFAR (with Context)',
                   color='#FF8C00', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add delta annotations
    for i, (no_ctx, alfar) in enumerate(zip(no_ctx_accs, alfar_accs)):
        delta = alfar - no_ctx
        y_pos = max(no_ctx, alfar) + 2
        color = 'green' if delta > 0 else 'red'
        ax.annotate(f'{delta:+.1f}%',
                   xy=(x[i], y_pos),
                   ha='center',
                   fontsize=11,
                   fontweight='bold',
                   color=color)

    # Add arrow annotation for Misleading bucket showing the drop
    if len(results) >= 3 and 'Misleading' in results:
        misleading_delta = results['Misleading']['alfar_acc'] - results['Misleading']['no_context_acc']
        if misleading_delta < 0:
            ax.annotate('',
                       xy=(2, results['Misleading']['no_context_acc']),
                       xytext=(2, results['Misleading']['alfar_acc']),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=3))
            ax.text(2.35, (results['Misleading']['no_context_acc'] + results['Misleading']['alfar_acc'])/2,
                   f'Drop:\n{abs(misleading_delta):.1f}%',
                   fontsize=11, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

    # Styling
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Context Reliability Bucket', fontsize=14, fontweight='bold')
    ax.set_title(f'ALFAR Performance by Context Reliability\n{dataset_name.upper()} Dataset',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=12)
    ax.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)
    ax.set_ylim(0, max(max(no_ctx_accs), max(alfar_accs)) + 15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add interpretation text
    interpretation = (
        "Interpretation: ALFAR blindly amplifies retrieved context.\n"
        "• Corrective: ALFAR helps when context is correct\n"
        "• Misleading: ALFAR hurts when context lacks the answer (key finding)"
    )
    plt.figtext(0.5, 0.01, interpretation,
               ha='center', fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as PDF
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    print(f"Figure saved to:")
    print(f"  - {output_path.with_suffix('.pdf')}")
    print(f"  - {output_path.with_suffix('.png')}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot bucket analysis results")
    parser.add_argument("--stats-file", type=str, required=True,
                       help="Path to bucket stats JSON file")
    parser.add_argument("--output-file", type=str, required=True,
                       help="Output file path (without extension)")
    parser.add_argument("--dataset-name", type=str, required=True,
                       help="Dataset display name")

    args = parser.parse_args()
    plot_bucket_results(args.stats_file, args.output_file, args.dataset_name)
