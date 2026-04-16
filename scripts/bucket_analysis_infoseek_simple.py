#!/usr/bin/env python3
"""
Simplified Three-Way Bucket Analysis for InfoSeek (MC task)
WITHOUT loading the 9.4GB knowledge base - uses prediction-based classification
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_results_jsonl(file_path):
    """Load JSONL results"""
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                results[item['data_id']] = item['prediction']
    return results


def load_ground_truth(question_file):
    """Load ground truth for InfoSeek"""
    with open(question_file, 'r') as f:
        questions = json.load(f)

    gt = {}
    for q in questions:
        gt[q['id']] = {
            'answer': q['multiple_choices_answer'],
            'choices': q['multiple_choices'],
            'question': q['question']
        }
    return gt


def evaluate_mc(pred, gold_answer):
    """Evaluate MC prediction"""
    if pred is None or pred == '':
        return False
    # Extract first character as prediction
    pred = str(pred).strip()[0].upper() if pred else ''
    return pred == gold_answer


def classify_sample_simplified(no_ctx_correct, regular_mrag_correct, alfar_correct):
    """
    Simplified bucket classification based on performance patterns.

    Since we don't have context quality labels, we classify based on:
    - Corrective: No-context wrong, at least one RAG method right
    - Neutral: No-context correct
    - Hard: All methods wrong (context insufficient)
    """
    if no_ctx_correct:
        return "Neutral"  # Model already knows
    elif regular_mrag_correct or alfar_correct:
        return "Corrective"  # Context helped at least one method
    else:
        return "Hard"  # Neither context method helped


def run_simplified_bucket_analysis(args):
    """Simplified three-way bucket analysis for InfoSeek"""

    # Load results
    print(f"Loading results for InfoSeek...")
    no_ctx_results = load_results_jsonl(args.no_context_file)
    regular_mrag_results = load_results_jsonl(args.regular_mrag_file)
    alfar_results = load_results_jsonl(args.alfar_file)
    ground_truth = load_ground_truth(args.question_file)

    print(f"Loaded {len(no_ctx_results)} no-context results")
    print(f"Loaded {len(regular_mrag_results)} regular MRAG results")
    print(f"Loaded {len(alfar_results)} ALFAR results")
    print(f"Loaded {len(ground_truth)} ground truth questions")

    # Find common question IDs
    common_ids = set(no_ctx_results.keys()) & set(regular_mrag_results.keys()) & set(alfar_results.keys())
    print(f"\nAnalyzing {len(common_ids)} common samples")

    # Classify samples into buckets
    buckets = {"Corrective": [], "Neutral": [], "Hard": []}
    bucket_assignments = {}

    for qid in common_ids:
        if qid not in ground_truth:
            continue

        gt_data = ground_truth[qid]
        gold_answer = gt_data['answer']

        # Evaluate predictions
        no_ctx_pred = no_ctx_results.get(qid, '')
        regular_mrag_pred = regular_mrag_results.get(qid, '')
        alfar_pred = alfar_results.get(qid, '')

        no_ctx_correct = evaluate_mc(no_ctx_pred, gold_answer)
        regular_mrag_correct = evaluate_mc(regular_mrag_pred, gold_answer)
        alfar_correct = evaluate_mc(alfar_pred, gold_answer)

        # Classify into bucket
        bucket = classify_sample_simplified(no_ctx_correct, regular_mrag_correct, alfar_correct)

        sample_info = {
            "question_id": qid,
            "question": gt_data['question'],
            "no_context_correct": no_ctx_correct,
            "regular_mrag_correct": regular_mrag_correct,
            "alfar_correct": alfar_correct,
            "no_context_pred": no_ctx_pred,
            "regular_mrag_pred": regular_mrag_pred,
            "alfar_pred": alfar_pred,
            "gold_answer": gold_answer,
            "choices": gt_data['choices']
        }

        buckets[bucket].append(sample_info)
        bucket_assignments[qid] = bucket

    # Compute statistics
    print("\n" + "="*80)
    print("SIMPLIFIED THREE-WAY BUCKET ANALYSIS RESULTS - INFOSEEK")
    print("="*80)

    results = {}
    for bucket_name in ["Corrective", "Neutral", "Hard"]:
        samples = buckets[bucket_name]
        if len(samples) == 0:
            print(f"\n{bucket_name}: No samples")
            continue

        no_ctx_acc = np.mean([s["no_context_correct"] for s in samples]) * 100
        regular_mrag_acc = np.mean([s["regular_mrag_correct"] for s in samples]) * 100
        alfar_acc = np.mean([s["alfar_correct"] for s in samples]) * 100

        results[bucket_name] = {
            "count": len(samples),
            "no_context_acc": no_ctx_acc,
            "regular_mrag_acc": regular_mrag_acc,
            "alfar_acc": alfar_acc,
            "delta_regular_mrag": regular_mrag_acc - no_ctx_acc,
            "delta_alfar": alfar_acc - no_ctx_acc,
            "delta_alfar_vs_regular": alfar_acc - regular_mrag_acc
        }

        print(f"\n{bucket_name} Bucket ({len(samples)} samples):")
        print(f"  No-Context Accuracy:     {no_ctx_acc:.2f}%")
        print(f"  Regular MRAG Accuracy:   {regular_mrag_acc:.2f}%")
        print(f"  ALFAR Accuracy:          {alfar_acc:.2f}%")
        print(f"  ───────────────────────────────────")
        print(f"  Delta (Regular MRAG - No-Ctx):      {regular_mrag_acc - no_ctx_acc:+.2f}%")
        print(f"  Delta (ALFAR - No-Ctx):             {alfar_acc - no_ctx_acc:+.2f}%")
        print(f"  Delta (ALFAR - Regular MRAG):       {alfar_acc - regular_mrag_acc:+.2f}%")

    # Overall accuracy
    all_samples = buckets["Corrective"] + buckets["Neutral"] + buckets["Hard"]
    print("\n" + "="*80)
    print("OVERALL ACCURACY")
    print("="*80)
    overall_no_ctx = np.mean([s["no_context_correct"] for s in all_samples]) * 100
    overall_regular = np.mean([s["regular_mrag_correct"] for s in all_samples]) * 100
    overall_alfar = np.mean([s["alfar_correct"] for s in all_samples]) * 100
    print(f"No-Context:    {overall_no_ctx:.2f}%")
    print(f"Regular MRAG:  {overall_regular:.2f}% ({overall_regular - overall_no_ctx:+.2f}%)")
    print(f"ALFAR:         {overall_alfar:.2f}% ({overall_alfar - overall_regular:+.2f}% vs Regular MRAG)")

    # Save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save bucket assignments
    with open(output_dir / 'infoseek_bucket_assignments_simple.json', 'w') as f:
        json.dump(bucket_assignments, f, indent=2)

    # Save bucket samples
    with open(output_dir / 'infoseek_bucket_samples_simple.json', 'w') as f:
        json.dump(buckets, f, indent=2)

    # Save summary statistics
    with open(output_dir / 'infoseek_bucket_stats_simple.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - infoseek_bucket_assignments_simple.json")
    print(f"  - infoseek_bucket_samples_simple.json")
    print(f"  - infoseek_bucket_stats_simple.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified three-way bucket analysis for InfoSeek")
    parser.add_argument("--no-context-file", type=str, required=True,
                       help="Path to no-context results file")
    parser.add_argument("--regular-mrag-file", type=str, required=True,
                       help="Path to regular MRAG results file")
    parser.add_argument("--alfar-file", type=str, required=True,
                       help="Path to ALFAR results file")
    parser.add_argument("--question-file", type=str, required=True,
                       help="Path to question data file")
    parser.add_argument("--output-dir", type=str, default="results/bucket_analysis_threeway",
                       help="Output directory for analysis results")

    args = parser.parse_args()
    run_simplified_bucket_analysis(args)
