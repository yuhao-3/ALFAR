#!/usr/bin/env python3
"""Calculate average accuracies across seeds for each dataset and method"""
import numpy as np

# Extracted accuracies from completed experiments
results = {
    'A-OKVQA': {
        'ALFAR': [0.6023, 0.6006, 0.6049, 0.5965, 0.6020],
        'TCVM': [0.5971, 0.6012, 0.5948, 0.5927, 0.5974]
    },
    'OKVQA': {
        'ALFAR': [0.6093, 0.6126, 0.6105, 0.6128, 0.6121],
        'TCVM': [0.6066, 0.6008, 0.6040, 0.6046, 0.6064]
    },
    'InfoSeek': {
        'ALFAR': [0.5730, 0.5713, 0.5737, 0.5750, 0.5783],
        'TCVM': [0.5747, 0.5720, 0.5743, 0.5717, 0.5797]
    }
}

print("=" * 80)
print("MULTISEED EXPERIMENT RESULTS (5 seeds: 0-4)")
print("=" * 80)
print()

# Print detailed results
for dataset, methods in results.items():
    print(f"{dataset}:")
    print("-" * 60)
    for method, accuracies in methods.items():
        print(f"  {method}:")
        for seed, acc in enumerate(accuracies):
            print(f"    Seed {seed}: {acc:.4f} ({acc*100:.2f}%)")
        mean = np.mean(accuracies)
        std = np.std(accuracies, ddof=1)  # Sample standard deviation
        print(f"    Mean: {mean:.4f} ± {std:.4f} ({mean*100:.2f}% ± {std*100:.2f}%)")
    print()

# Summary table
print("=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print()
print(f"{'Dataset':<15} {'ALFAR Mean':<20} {'TCVM Mean':<20} {'Difference':<15}")
print("-" * 80)

for dataset, methods in results.items():
    alfar_mean = np.mean(methods['ALFAR'])
    alfar_std = np.std(methods['ALFAR'], ddof=1)
    tcvm_mean = np.mean(methods['TCVM'])
    tcvm_std = np.std(methods['TCVM'], ddof=1)
    diff = alfar_mean - tcvm_mean

    print(f"{dataset:<15} {alfar_mean:.4f} ± {alfar_std:.4f}    {tcvm_mean:.4f} ± {tcvm_std:.4f}    {diff:+.4f}")

print()
print("=" * 80)
print("NOTES:")
print("- Mean calculated across 5 seeds (0-4)")
print("- ± shows sample standard deviation")
print("- Difference = ALFAR - TCVM (positive means ALFAR is better)")
print("=" * 80)
