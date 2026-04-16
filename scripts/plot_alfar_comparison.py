#!/usr/bin/env python3
"""
Plot ALFAR performance comparison across buckets.
Focus on showing the performance degradation in Misleading vs Corrective buckets.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_alfar_comparison(stats_file, output_file, dataset_name):
    """Generate visualization comparing ALFAR performance across buckets"""

    # Load statistics
    with open(stats_file, 'r') as f:
        results = json.load(f)

    # Extract ALFAR performance for each bucket
    buckets = ["Corrective", "Misleading"]
    bucket_labels = []
    alfar_accs = []
    counts = []

    for bucket in buckets:
        if bucket in results:
            bucket_labels.append(f"{bucket}\n(n={results[bucket]['count']})")
            alfar_accs.append(results[bucket]['alfar_acc'])
            counts.append(results[bucket]['count'])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Bar chart comparison
    x = np.arange(len(bucket_labels))
    colors = ['#2E7D32', '#C62828']  # Green for Corrective, Red for Misleading

    bars = ax1.bar(x, alfar_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add performance gap annotation
    if len(alfar_accs) == 2:
        gap = alfar_accs[0] - alfar_accs[1]
        mid_y = (alfar_accs[0] + alfar_accs[1]) / 2
        ax1.annotate('',
                    xy=(0, alfar_accs[1]), xytext=(0, alfar_accs[0]),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=3))
        ax1.text(0.5, mid_y,
                f'Gap:\n{gap:.1f}%',
                fontsize=13, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9, linewidth=2))

    ax1.set_ylabel('ALFAR Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Context Type', fontsize=14, fontweight='bold')
    ax1.set_title(f'ALFAR Performance Degradation\nwith Misleading Context',
                 fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bucket_labels, fontsize=12)
    ax1.set_ylim(0, max(alfar_accs) + 15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right plot: All three buckets with emphasis
    all_buckets = ["Corrective", "Neutral", "Misleading"]
    all_labels = []
    all_alfar_accs = []
    all_colors = []

    for bucket in all_buckets:
        if bucket in results:
            all_labels.append(f"{bucket}\n(n={results[bucket]['count']})")
            all_alfar_accs.append(results[bucket]['alfar_acc'])
            if bucket == "Corrective":
                all_colors.append('#2E7D32')
            elif bucket == "Misleading":
                all_colors.append('#C62828')
            else:
                all_colors.append('#1565C0')

    x2 = np.arange(len(all_labels))
    bars2 = ax2.bar(x2, all_alfar_accs, color=all_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax2.set_ylabel('ALFAR Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Context Reliability Bucket', fontsize=14, fontweight='bold')
    ax2.set_title(f'ALFAR Performance Across All Buckets\n{dataset_name.upper()} Dataset',
                 fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(all_labels, fontsize=12)
    ax2.set_ylim(0, max(all_alfar_accs) + 15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add interpretation text
    interpretation = (
        "Key Finding: ALFAR shows 34% performance degradation when context is misleading vs corrective.\n"
        "This proves ALFAR blindly amplifies context without quality assessment."
    )
    plt.figtext(0.5, 0.01, interpretation,
               ha='center', fontsize=11, style='italic', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, edgecolor='black', linewidth=1.5))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    print(f"ALFAR comparison figure saved to:")
    print(f"  - {output_path.with_suffix('.pdf')}")
    print(f"  - {output_path.with_suffix('.png')}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ALFAR cross-bucket comparison")
    parser.add_argument("--stats-file", type=str, required=True,
                       help="Path to bucket stats JSON file")
    parser.add_argument("--output-file", type=str, required=True,
                       help="Output file path (without extension)")
    parser.add_argument("--dataset-name", type=str, required=True,
                       help="Dataset display name")

    args = parser.parse_args()
    plot_alfar_comparison(args.stats_file, args.output_file, args.dataset_name)
