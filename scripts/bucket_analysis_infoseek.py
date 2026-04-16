#!/usr/bin/env python3
"""
Three-Way Bucket Analysis for InfoSeek (MC task)

MC tasks have cleaner answer matching than open-ended tasks.
Bucket classification: check if correct option text appears in context.
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


def load_contexts(question_file):
    """Load retrieved contexts for InfoSeek (memory-optimized)"""
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / 'data'

    # Load questions
    with open(question_file, 'r') as f:
        questions = json.load(f)

    print(f"Loading retrieval indices...")
    # Load retrieval info
    indices = np.load(DATA_DIR / 'retrieval_result' / 'infoseek_mc_indices_50_17k.npy', allow_pickle=True)
    indice_map = np.load(DATA_DIR / 'wiki' / 'wiki_map_17k.npy', allow_pickle=True).item()

    print(f"Loading knowledge base (this may take a minute)...")
    # Load knowledge base - allow pickle for dict
    knowledge_base = np.load(DATA_DIR / 'wiki' / 'wiki_with_image.npy', allow_pickle=True).item()

    print(f"Extracting contexts for {len(questions)} questions...")
    contexts = {}
    for i, q in enumerate(questions):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(questions)} questions...")
        indice = indices[i]
        know_index = indice_map[indice[0]]
        context = knowledge_base[know_index]['wikipedia_summary']
        contexts[q['id']] = context

    print(f"Loaded contexts for {len(contexts)} questions")
    # Clear knowledge_base from memory
    del knowledge_base
    return contexts


def check_context_support_mc(context_text, gold_answer, question_data):
    """
    Check if retrieved context contains the correct answer for MC task.
    
    For MC: gold_answer is option letter (A/B/C/D), get actual text from choices.
    """
    if context_text is None or context_text == '':
        return False

    context_lower = context_text.lower()

    # MC task - get option text from answer letter
    if gold_answer not in ['A', 'B', 'C', 'D']:
        return False
    
    # Get the actual text of the correct option
    correct_option_text = question_data['choices'][gold_answer]
    
    # Check if answer text appears in context
    return correct_option_text.lower() in context_lower


def classify_sample_mc(question_id, no_ctx_correct, context_text, gold_answer, question_data):
    """
    Classify each sample into one of three buckets for MC task.

    - Corrective: Model wrong without context, context has answer
    - Neutral: Model correct without context, context has answer
    - Misleading: Model wrong without context, context lacks answer
    - Other: Model correct without context, context contradicts (discard)
    """
    context_has_answer = check_context_support_mc(context_text, gold_answer, question_data)

    if not no_ctx_correct and context_has_answer:
        return "Corrective"      # Context can help, model needs it
    elif no_ctx_correct and context_has_answer:
        return "Neutral"         # Model already knows, context confirms
    elif not no_ctx_correct and not context_has_answer:
        return "Misleading"      # Context cannot help, may mislead
    else:
        return "Other"           # Model correct but context contradicts - discard


def evaluate_mc(pred, gold_answer):
    """Evaluate MC prediction"""
    if pred is None or pred == '':
        return False
    # Extract first character as prediction
    pred = str(pred).strip()[0].upper() if pred else ''
    return pred == gold_answer


def run_threeway_bucket_analysis(args):
    """Main three-way bucket analysis logic for InfoSeek"""

    # Load results
    print(f"Loading results for InfoSeek...")
    no_ctx_results = load_results_jsonl(args.no_context_file)
    regular_mrag_results = load_results_jsonl(args.regular_mrag_file)
    alfar_results = load_results_jsonl(args.alfar_file)
    ground_truth = load_ground_truth(args.question_file)
    contexts = load_contexts(args.question_file)

    print(f"Loaded {len(no_ctx_results)} no-context results")
    print(f"Loaded {len(regular_mrag_results)} regular MRAG results")
    print(f"Loaded {len(alfar_results)} ALFAR results")
    print(f"Loaded {len(ground_truth)} ground truth questions")

    # Classify samples into buckets
    buckets = {"Corrective": [], "Neutral": [], "Misleading": []}
    bucket_assignments = {}

    for qid in ground_truth.keys():
        if qid not in no_ctx_results or qid not in regular_mrag_results or qid not in alfar_results:
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

        # Get context
        context = contexts.get(qid, '')

        # Classify into bucket based on NO-CONTEXT performance
        bucket = classify_sample_mc(qid, no_ctx_correct, context, gold_answer, gt_data)

        if bucket == "Other":
            continue

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
            "choices": gt_data['choices'],
            "context_preview": context[:200] + "..." if len(context) > 200 else context
        }

        buckets[bucket].append(sample_info)
        bucket_assignments[qid] = bucket

    # Compute statistics
    print("\n" + "="*80)
    print("THREE-WAY BUCKET ANALYSIS RESULTS - INFOSEEK")
    print("="*80)

    results = {}
    for bucket_name in ["Corrective", "Neutral", "Misleading"]:
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

    # Special analysis for Misleading bucket
    if "Misleading" in results and results["Misleading"]["count"] > 0:
        print("\n" + "="*80)
        print("CRITICAL ANALYSIS: Misleading Bucket")
        print("="*80)

        misleading_stats = results["Misleading"]
        regular_vs_alfar = misleading_stats["delta_alfar_vs_regular"]

        print(f"\nIn Misleading bucket (context lacks answer):")
        print(f"  Regular MRAG: {misleading_stats['regular_mrag_acc']:.2f}%")
        print(f"  ALFAR:        {misleading_stats['alfar_acc']:.2f}%")
        print(f"  Difference:   {regular_vs_alfar:+.2f}%")

        print("\nInterpretation:")
        if regular_vs_alfar < -2.0:  # ALFAR significantly worse
            print("  ❌ ALFAR amplification CAUSES ADDITIONAL HARM")
            print("  → ALFAR's amplification of misleading context makes performance worse")
        elif abs(regular_vs_alfar) <= 2.0:  # Similar
            print("  ⚠️  Performance is similar")
            print("  → Problem is misleading context itself, not ALFAR's amplification")
        else:  # ALFAR better
            print("  ✅ ALFAR performs BETTER despite misleading context")
            print("  → ALFAR's mechanisms help filter/ignore misleading information")

    # Save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save bucket assignments
    with open(output_dir / 'infoseek_bucket_assignments.json', 'w') as f:
        json.dump(bucket_assignments, f, indent=2)

    # Save bucket samples
    with open(output_dir / 'infoseek_bucket_samples.json', 'w') as f:
        json.dump(buckets, f, indent=2)

    # Save summary statistics
    with open(output_dir / 'infoseek_bucket_stats.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - infoseek_bucket_assignments.json")
    print(f"  - infoseek_bucket_samples.json")
    print(f"  - infoseek_bucket_stats.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Three-way bucket analysis for InfoSeek")
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
    run_threeway_bucket_analysis(args)
