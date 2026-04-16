#!/usr/bin/env python3
"""
Plot three-way bucket analysis results for InfoSeek (MC task).

Shows comparison between:
1. No-Context (Parametric)
2. Regular MRAG (Standard RAG)
3. ALFAR (ALFAR with Context)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_threeway_bucket_analysis(dataset_name, stats_file, output_file):
    """Create bar chart comparing three methods across buckets"""

    # Load statistics
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    # Prepare data
    buckets = ["Corrective", "Neutral", "Misleading"]
    no_ctx_accs = []
    regular_mrag_accs = []
    alfar_accs = []
    counts = []

    for bucket in buckets:
        if bucket in stats:
            no_ctx_accs.append(stats[bucket]['no_context_acc'])
            regular_mrag_accs.append(stats[bucket]['regular_mrag_acc'])
            alfar_accs.append(stats[bucket]['alfar_acc'])
            counts.append(stats[bucket]['count'])
        else:
            no_ctx_accs.append(0)
            regular_mrag_accs.append(0)
            alfar_accs.append(0)
            counts.append(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(buckets))
    width = 0.25

    # Create bars
    bars1 = ax.bar(x - width, no_ctx_accs, width, label='No-Context',
                   color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x, regular_mrag_accs, width, label='Regular MRAG',
                   color='#F39C12', alpha=0.8)
    bars3 = ax.bar(x + width, alfar_accs, width, label='ALFAR',
                   color='#27AE60', alpha=0.8)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    # Add sample counts below bucket names
    bucket_labels = [f"{bucket}\n(n={count})" for bucket, count in zip(buckets, counts)]
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=12)

    # Labels and title
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Context Quality Bucket', fontsize=13, fontweight='bold')
    ax.set_title(f'Three-Way Bucket Analysis: {dataset_name.upper()}\nNo-Context vs Regular MRAG vs ALFAR',
                fontsize=14, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 110)

    # Add interpretation text
    if "Misleading" in stats:
        misleading_stats = stats["Misleading"]
        delta = misleading_stats['delta_alfar_vs_regular']

        interpretation = f"Critical Finding (Misleading Bucket):\n"
        if delta < -2.0:
            interpretation += f"❌ ALFAR amplification causes harm ({delta:+.2f}%)"
            color = 'red'
        elif abs(delta) <= 2.0:
            interpretation += f"⚠️  Similar performance ({delta:+.2f}%)"
            color = 'orange'
        else:
            interpretation += f"✅ ALFAR filters misleading context ({delta:+.2f}%)"
            color = 'green'

        ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
               fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


def plot_delta_comparison(dataset_name, stats_file, output_file):
    """Plot delta changes from No-Context baseline"""

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    buckets = ["Corrective", "Neutral", "Misleading"]
    regular_mrag_deltas = []
    alfar_deltas = []

    for bucket in buckets:
        if bucket in stats:
            regular_mrag_deltas.append(stats[bucket]['delta_regular_mrag'])
            alfar_deltas.append(stats[bucket]['delta_alfar'])
        else:
            regular_mrag_deltas.append(0)
            alfar_deltas.append(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(buckets))
    width = 0.35

    bars1 = ax.bar(x - width/2, regular_mrag_deltas, width,
                   label='Regular MRAG (vs No-Context)', color='#F39C12', alpha=0.8)
    bars2 = ax.bar(x + width/2, alfar_deltas, width,
                   label='ALFAR (vs No-Context)', color='#27AE60', alpha=0.8)

    # Add value labels
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}%',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10, fontweight='bold')

    autolabel(bars1)
    autolabel(bars2)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=12)
    ax.set_ylabel('Accuracy Change (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Context Quality Bucket', fontsize=13, fontweight='bold')
    ax.set_title(f'Performance Delta from No-Context Baseline: {dataset_name.upper()}',
                fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    # InfoSeek results
    dataset = "infoseek"
    stats_file = Path("results/bucket_analysis_threeway/infoseek_bucket_stats.json")

    # Create output directory
    output_dir = Path("results/bucket_analysis_threeway")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_threeway_bucket_analysis(
        dataset,
        stats_file,
        output_dir / f"{dataset}_threeway_comparison.png"
    )

    plot_delta_comparison(
        dataset,
        stats_file,
        output_dir / f"{dataset}_delta_comparison.png"
    )

    print("All InfoSeek plots generated successfully!")
