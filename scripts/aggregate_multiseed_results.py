#!/usr/bin/env python3
"""
Aggregate results from multiple seed runs and compute mean ± std.

Usage:
    python scripts/aggregate_multiseed_results.py \
        --dataset okvqa \
        --method tcvm \
        --seeds 0 1 2 3 4 \
        --results_dir experiments/result/multiseed
"""

import argparse
import json
import re
import numpy as np
from pathlib import Path


def parse_metric_file(metric_file):
    """
    Parse evaluation metrics from a text file.

    Expected format:
        Accuracy: 0.5234
        F1 Score: 0.6123
        ...

    Returns:
        dict: {metric_name: value}
    """
    metrics = {}

    with open(metric_file, 'r') as f:
        content = f.read()

        # Common patterns for metrics
        patterns = {
            'accuracy': r'[Aa]ccuracy[:\s]+([0-9.]+)',
            'f1': r'[Ff]1[:\s]+([0-9.]+)',
            'precision': r'[Pp]recision[:\s]+([0-9.]+)',
            'recall': r'[Rr]ecall[:\s]+([0-9.]+)',
            'exact_match': r'[Ee]xact[_\s][Mm]atch[:\s]+([0-9.]+)',
        }

        for metric_name, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[metric_name] = float(match.group(1))

    return metrics


def aggregate_results(dataset, method, seeds, results_dir):
    """
    Aggregate results from multiple seeds.

    Args:
        dataset: Dataset name (e.g., 'okvqa', 'aokvqa', 'infoseek')
        method: Method name (e.g., 'tcvm', 'alfar')
        seeds: List of seed values
        results_dir: Directory containing result files

    Returns:
        dict: {metric_name: {'mean': mean, 'std': std, 'values': [...]}}
    """
    results_dir = Path(results_dir)
    logs_dir = Path('logs')

    all_metrics = {}

    for seed in seeds:
        # Try to find metric file
        metric_file = logs_dir / f"{method}_{dataset}_seed{seed}_metrics.txt"

        if not metric_file.exists():
            print(f"Warning: Metric file not found for seed {seed}: {metric_file}")
            continue

        # Parse metrics
        metrics = parse_metric_file(metric_file)

        # Aggregate
        for metric_name, value in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    # Compute statistics
    stats = {}
    for metric_name, values in all_metrics.items():
        if len(values) == 0:
            continue

        stats[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
            'values': values,
            'n': len(values)
        }

    return stats


def print_statistics(stats, dataset, method):
    """Print statistics in a nice format."""
    print("\n" + "=" * 60)
    print(f"Results for {method.upper()} on {dataset.upper()}")
    print("=" * 60)

    if not stats:
        print("No metrics found!")
        return

    print(f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'N':<5} {'Values'}")
    print("-" * 60)

    for metric_name, stat in sorted(stats.items()):
        values_str = ', '.join([f"{v:.4f}" for v in stat['values']])
        print(f"{metric_name:<20} {stat['mean']:.4f}     ±{stat['std']:.4f}     {stat['n']:<5} [{values_str}]")

    print("\n" + "=" * 60)
    print("LaTeX-friendly format:")
    print("-" * 60)

    for metric_name, stat in sorted(stats.items()):
        print(f"{metric_name}: ${stat['mean']*100:.2f} \\pm {stat['std']*100:.2f}$")

    print("=" * 60 + "\n")


def save_statistics(stats, output_file):
    """Save statistics to JSON file."""
    output = {}

    for metric_name, stat in stats.items():
        output[metric_name] = {
            'mean': float(stat['mean']),
            'std': float(stat['std']),
            'n': int(stat['n']),
            'values': [float(v) for v in stat['values']]
        }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Statistics saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate multi-seed evaluation results'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['okvqa', 'aokvqa', 'infoseek', 'viquae', 'evqa'],
        help='Dataset name'
    )
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['tcvm', 'alfar', 'vcd', 'baseline'],
        help='Method name'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
        help='List of seed values (default: 0 1 2 3 4)'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='experiments/result/multiseed',
        help='Directory containing result files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for statistics (optional)'
    )

    args = parser.parse_args()

    # Aggregate results
    stats = aggregate_results(
        dataset=args.dataset,
        method=args.method,
        seeds=args.seeds,
        results_dir=args.results_dir
    )

    # Print statistics
    print_statistics(stats, args.dataset, args.method)

    # Save statistics if output file specified
    if args.output:
        save_statistics(stats, args.output)
    elif stats:
        # Auto-generate output filename
        output_file = f"experiments/result/{args.method}_{args.dataset}_stats.json"
        save_statistics(stats, output_file)


if __name__ == '__main__':
    main()
