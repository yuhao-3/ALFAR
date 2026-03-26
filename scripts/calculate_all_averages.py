#!/usr/bin/env python3
"""
Calculate average metrics across all seeds for all datasets and methods.
Generates comprehensive summary tables.
"""

import re
import numpy as np
from pathlib import Path
import sys


def parse_metric_file(metric_file):
    """Parse accuracy from a metrics file."""
    if not metric_file.exists():
        return None

    with open(metric_file, 'r') as f:
        content = f.read()

    # Look for accuracy pattern
    match = re.search(r'[Aa]ccuracy[:\s]+([0-9.]+)', content)
    if match:
        return float(match.group(1))

    return None


def collect_results(datasets, methods, seeds, logs_dir='logs'):
    """
    Collect all results from metrics files.

    Returns:
        dict: {dataset: {method: {seed: accuracy}}}
    """
    logs_path = Path(logs_dir)
    results = {}

    for dataset in datasets:
        results[dataset] = {}

        for method in methods:
            results[dataset][method] = {}

            for seed in seeds:
                metric_file = logs_path / f"{method}_{dataset}_seed{seed}_metrics.txt"
                accuracy = parse_metric_file(metric_file)

                if accuracy is not None:
                    results[dataset][method][seed] = accuracy

    return results


def calculate_statistics(values):
    """Calculate mean and std from a list of values."""
    if not values:
        return None, None

    values_array = np.array(values)
    mean = np.mean(values_array)
    std = np.std(values_array, ddof=1) if len(values_array) > 1 else 0.0

    return mean, std


def print_detailed_results(results, datasets, methods, seeds):
    """Print detailed results for each dataset and method."""
    print("=" * 80)
    print("MULTISEED EXPERIMENT RESULTS")
    print("=" * 80)
    print()

    for dataset in datasets:
        if dataset not in results:
            continue

        print(f"\n{'='*80}")
        print(f"{dataset.upper()}")
        print('='*80)

        for method in methods:
            if method not in results[dataset]:
                continue

            method_results = results[dataset][method]
            if not method_results:
                continue

            print(f"\n  {method.upper()}:")
            print("  " + "-" * 60)

            # Print individual seed results
            values = []
            for seed in seeds:
                if seed in method_results:
                    acc = method_results[seed]
                    values.append(acc)
                    print(f"    Seed {seed}: {acc:.4f} ({acc*100:.2f}%)")
                else:
                    print(f"    Seed {seed}: MISSING")

            # Print statistics
            if values:
                mean, std = calculate_statistics(values)
                print(f"    {'─'*56}")
                print(f"    Mean:   {mean:.4f} ± {std:.4f} ({mean*100:.2f}% ± {std*100:.2f}%)")
                print(f"    N:      {len(values)}/{len(seeds)} seeds")

        print()


def print_summary_table(results, datasets, methods):
    """Print summary table comparing methods."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Mean ± Std (across all seeds)")
    print("=" * 80)
    print()

    # Header
    print(f"{'Dataset':<15} {'ALFAR':<25} {'TCVM':<25} {'Diff':<10}")
    print("-" * 80)

    # Data rows
    for dataset in datasets:
        if dataset not in results:
            continue

        dataset_results = results[dataset]

        # Get ALFAR results
        alfar_values = []
        if 'alfar' in dataset_results:
            alfar_values = list(dataset_results['alfar'].values())
        alfar_mean, alfar_std = calculate_statistics(alfar_values)

        # Get TCVM results
        tcvm_values = []
        if 'tcvm' in dataset_results:
            tcvm_values = list(dataset_results['tcvm'].values())
        tcvm_mean, tcvm_std = calculate_statistics(tcvm_values)

        # Format output
        if alfar_mean is not None and tcvm_mean is not None:
            diff = alfar_mean - tcvm_mean
            alfar_str = f"{alfar_mean:.4f} ± {alfar_std:.4f}"
            tcvm_str = f"{tcvm_mean:.4f} ± {tcvm_std:.4f}"
            diff_str = f"{diff:+.4f}"

            # Add visual indicator
            if diff > 0:
                diff_str += " ✓"
            elif diff < 0:
                diff_str += " ✗"

            print(f"{dataset:<15} {alfar_str:<25} {tcvm_str:<25} {diff_str:<10}")
        elif alfar_mean is not None:
            alfar_str = f"{alfar_mean:.4f} ± {alfar_std:.4f}"
            print(f"{dataset:<15} {alfar_str:<25} {'N/A':<25} {'N/A':<10}")
        elif tcvm_mean is not None:
            tcvm_str = f"{tcvm_mean:.4f} ± {tcvm_std:.4f}"
            print(f"{dataset:<15} {'N/A':<25} {tcvm_str:<25} {'N/A':<10}")

    print("\n" + "=" * 80)
    print("Notes:")
    print("  - Diff = ALFAR - TCVM (positive means ALFAR is better)")
    print("  - ✓ = ALFAR better, ✗ = TCVM better")
    print("  - Std = Sample standard deviation")
    print("=" * 80)


def print_latex_table(results, datasets, methods):
    """Print LaTeX-formatted table."""
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l|c|c|c}")
    print("\\hline")
    print("Dataset & ALFAR & TCVM & Diff \\\\")
    print("\\hline")

    for dataset in datasets:
        if dataset not in results:
            continue

        dataset_results = results[dataset]

        # Get ALFAR results
        alfar_values = list(dataset_results.get('alfar', {}).values())
        alfar_mean, alfar_std = calculate_statistics(alfar_values)

        # Get TCVM results
        tcvm_values = list(dataset_results.get('tcvm', {}).values())
        tcvm_mean, tcvm_std = calculate_statistics(tcvm_values)

        if alfar_mean is not None and tcvm_mean is not None:
            diff = (alfar_mean - tcvm_mean) * 100
            print(f"{dataset.upper()} & "
                  f"${alfar_mean*100:.2f} \\pm {alfar_std*100:.2f}$ & "
                  f"${tcvm_mean*100:.2f} \\pm {tcvm_std*100:.2f}$ & "
                  f"${diff:+.2f}$ \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Comparison of ALFAR and TCVM across datasets (mean $\\pm$ std)}")
    print("\\end{table}")
    print("\n" + "=" * 80)


def print_missing_results(results, datasets, methods, seeds):
    """Print summary of missing results."""
    missing = []

    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                if dataset not in results or method not in results[dataset] or seed not in results[dataset][method]:
                    missing.append(f"{method}_{dataset}_seed{seed}")

    if missing:
        print("\n" + "=" * 80)
        print(f"MISSING RESULTS ({len(missing)} total)")
        print("=" * 80)
        for item in sorted(missing):
            print(f"  - {item}")
        print("=" * 80)


def main():
    # Configuration
    datasets = ['aokvqa', 'okvqa', 'infoseek', 'viquae', 'evqa']
    methods = ['alfar', 'tcvm']
    seeds = [0, 1, 2, 3, 4]

    print("Collecting results from logs directory...")
    results = collect_results(datasets, methods, seeds)

    # Print detailed results
    print_detailed_results(results, datasets, methods, seeds)

    # Print summary table
    print_summary_table(results, datasets, methods)

    # Print LaTeX table
    print_latex_table(results, datasets, methods)

    # Print missing results
    print_missing_results(results, datasets, methods, seeds)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
