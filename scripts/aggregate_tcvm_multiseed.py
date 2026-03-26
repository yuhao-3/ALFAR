#!/usr/bin/env python3
"""
Aggregate multi-seed TCVM-KAR results and calculate mean(sd)
Usage: python scripts/aggregate_tcvm_multiseed.py --model llava1.5 --dataset infoseek
"""

import argparse
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path

def evaluate_infoseek_viquae(result_file, dataset):
    """Evaluate InfoSeek or ViQuAE results"""
    correct = 0
    total = 0
    
    with open(result_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'correct' in data and data['correct']:
                correct += 1
            total += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, correct, total

def evaluate_okvqa_aokvqa(result_file, dataset):
    """Evaluate OK-VQA or A-OKVQA results"""
    df = pd.read_csv(result_file)
    
    if 'acc' in df.columns:
        accuracy = df['acc'].mean() * 100
        correct = df['acc'].sum()
        total = len(df)
    else:
        # Fallback: count correct answers
        accuracy = 0
        correct = 0
        total = len(df)
    
    return accuracy, correct, total

def evaluate_evqa(result_file):
    """Evaluate E-VQA results (just count lines for now, actual eval needs TensorFlow)"""
    # For E-VQA, we need to run the actual evaluation script
    # For now, we'll just return a placeholder
    import subprocess
    try:
        result = subprocess.run(
            ['python', 'evaluation/eval_evqa.py', '--preds', result_file],
            capture_output=True,
            text=True,
            timeout=300
        )
        # Parse accuracy from output
        for line in result.stdout.split('\n'):
            if 'Score:' in line or 'Accuracy:' in line:
                acc_str = line.split(':')[1].strip().split()[0]
                accuracy = float(acc_str)
                return accuracy, None, None
    except:
        pass
    
    return None, None, None

def aggregate_results(model, dataset, seeds=[0, 1, 2]):
    """Aggregate results across seeds"""
    
    results_dir = Path(f'results/{model}/{dataset}')
    
    # Determine file extension
    if dataset in ['infoseek', 'viquae']:
        ext = 'jsonl'
        eval_func = lambda f: evaluate_infoseek_viquae(f, dataset)
    elif dataset in ['aokvqa', 'okvqa']:
        ext = 'csv'
        eval_func = lambda f: evaluate_okvqa_aokvqa(f, dataset)
    elif dataset == 'evqa':
        ext = 'json'
        eval_func = evaluate_evqa
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    accuracies = []
    
    for seed in seeds:
        result_file = results_dir / f'{dataset}_tcvm_seed{seed}_results.{ext}'
        
        if not result_file.exists():
            print(f"Warning: {result_file} not found, skipping seed {seed}")
            continue
        
        print(f"Evaluating seed {seed}: {result_file}")
        
        accuracy, correct, total = eval_func(result_file)
        
        if accuracy is not None:
            accuracies.append(accuracy)
            print(f"  Seed {seed}: {accuracy:.2f}%")
    
    if len(accuracies) == 0:
        print(f"ERROR: No results found for {dataset}")
        return None
    
    # Calculate mean and std
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0
    
    print(f"\n{'='*50}")
    print(f"{dataset.upper()} Results (n={len(accuracies)})")
    print(f"{'='*50}")
    print(f"Individual seeds: {[f'{a:.2f}%' for a in accuracies]}")
    print(f"Mean (SD): {mean_acc:.2f}% ({std_acc:.2f}%)")
    print(f"{'='*50}\n")
    
    # Save summary
    summary_file = results_dir / 'accuracy_multiseed_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Model: {model}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Method: TCVM-KAR\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Config: topk=20, alpha=1.0, beta=0.7\n")
        f.write(f"\n")
        f.write(f"Individual Results:\n")
        for i, (seed, acc) in enumerate(zip(seeds[:len(accuracies)], accuracies)):
            f.write(f"  Seed {seed}: {acc:.2f}%\n")
        f.write(f"\n")
        f.write(f"Mean (SD): {mean_acc:.2f}% ({std_acc:.2f}%)\n")
    
    print(f"Summary saved to: {summary_file}")
    
    return {
        'dataset': dataset,
        'mean': mean_acc,
        'std': std_acc,
        'seeds': accuracies,
        'n': len(accuracies)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate multi-seed TCVM-KAR results')
    parser.add_argument('--model', type=str, default='llava1.5', help='Model name')
    parser.add_argument('--dataset', type=str, help='Dataset name (infoseek, viquae, aokvqa, okvqa, evqa)')
    parser.add_argument('--all', action='store_true', help='Aggregate all datasets')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Seeds to aggregate')
    
    args = parser.parse_args()
    
    if args.all:
        datasets = ['infoseek', 'viquae', 'aokvqa', 'okvqa', 'evqa']
        all_results = []
        
        print(f"\n{'='*70}")
        print(f"AGGREGATING ALL DATASETS FOR {args.model.upper()}")
        print(f"{'='*70}\n")
        
        for dataset in datasets:
            result = aggregate_results(args.model, dataset, args.seeds)
            if result:
                all_results.append(result)
        
        # Print final summary table
        print(f"\n{'='*70}")
        print(f"FINAL SUMMARY - {args.model.upper()} TCVM-KAR")
        print(f"{'='*70}")
        print(f"{'Dataset':<12} {'Mean':>10} {'SD':>8} {'Individual Seeds':<30}")
        print(f"{'-'*70}")
        
        for r in all_results:
            seeds_str = ', '.join([f'{s:.2f}%' for s in r['seeds']])
            print(f"{r['dataset']:<12} {r['mean']:>9.2f}% {r['std']:>7.2f}% {seeds_str}")
        
        print(f"{'='*70}\n")
        
    elif args.dataset:
        aggregate_results(args.model, args.dataset, args.seeds)
    else:
        parser.print_help()

