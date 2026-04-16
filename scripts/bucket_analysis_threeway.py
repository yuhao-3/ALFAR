#!/usr/bin/env python3
"""
Three-Way Bucket Analysis for Motivation Experiment

Compares three methods:
1. No-Context (Parametric): No context + No ALFAR
2. Regular MRAG (Standard RAG): Has context + No ALFAR
3. ALFAR: Has context + Has ALFAR

Classifies samples into three buckets based on context reliability:
- Corrective: Context helps (model wrong without it, context has answer)
- Neutral: Context confirms (model already correct, context has answer)
- Misleading: Context may mislead (model wrong without it, context lacks answer)

Then computes accuracy for each method within each bucket.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_results_jsonl(file_path):
    """Load JSONL results (InfoSeek/ViQuAE format)"""
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            results[item['data_id']] = item['prediction']
    return results


def load_results_csv(file_path):
    """Load CSV results (OKVQA/AOKVQA format)"""
    df = pd.read_csv(file_path)
    results = {}
    for _, row in df.iterrows():
        results[str(row['question_id'])] = row['llama_answer']
    return results


def load_ground_truth_mc(question_file):
    """Load ground truth for MC datasets (InfoSeek/ViQuAE)"""
    with open(question_file, 'r') as f:
        questions = json.load(f)

    gt = {}
    for q in questions:
        gt[q['id']] = {
            'answer': q['answer'],
            'choices': q['multiple_choices'],
            'question': q['question']
        }
    return gt


def load_ground_truth_okvqa(csv_file):
    """Load ground truth for OKVQA/AOKVQA"""
    df = pd.read_csv(csv_file)
    gt = {}
    for _, row in df.iterrows():
        # Parse answers (assuming format like "['answer1', 'answer2']")
        answers_str = row['answers']
        if isinstance(answers_str, str):
            answers = eval(answers_str)  # Convert string representation to list
        else:
            answers = [str(answers_str)]

        gt[str(row['question_id'])] = {
            'answer': answers,
            'question': row['question']
        }
    return gt


def load_contexts_mc(question_file, dataset):
    """Load retrieved contexts for MC datasets"""
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / 'data'

    # Load questions
    with open(question_file, 'r') as f:
        questions = json.load(f)

    # Load retrieval info
    if dataset == 'infoseek':
        indices = np.load(DATA_DIR / 'retrieval_result' / 'infoseek_mc_indices_50_17k.npy', allow_pickle=True)
        indice_map = np.load(DATA_DIR / 'wiki' / 'wiki_map_17k.npy', allow_pickle=True).item()
    elif dataset == 'viquae':
        indices = np.load(DATA_DIR / 'retrieval_result' / 'viquae_indices_50.npy', allow_pickle=True)
        indice_map = np.load(DATA_DIR / 'wiki' / 'wiki_map.npy', allow_pickle=True).item()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    knowledge_base = np.load(DATA_DIR / 'wiki' / 'wiki_with_image.npy', allow_pickle=True).item()

    contexts = {}
    for i, q in enumerate(questions):
        indice = indices[i]
        know_index = indice_map[indice[0]]
        context = knowledge_base[know_index]['wikipedia_summary']
        contexts[q['id']] = context

    return contexts


def load_contexts_okvqa(dataset):
    """Load retrieved contexts for OKVQA/AOKVQA"""
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / 'data' / 'eval_data' / 'okvqa'

    if dataset == 'aokvqa':
        with open(DATA_DIR / 'aokvqa_val_dcaption.json', 'r') as f:
            knowledge = json.load(f)
    elif dataset == 'okvqa':
        with open(DATA_DIR / 'okvqa_val_dcaption.json', 'r') as f:
            knowledge = json.load(f)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return knowledge


def check_context_support(context_text, gold_answer, question_data, is_mc=True):
    """
    Check if retrieved context contains information supporting the gold answer.

    For MC datasets: gold_answer is option letter (A/B/C/D), get actual text from choices
    For open-ended: gold_answer is list of answer strings
    """
    if context_text is None or context_text == '':
        return False

    context_lower = context_text.lower()

    if is_mc:
        # MC task - get option text from answer letter
        if gold_answer not in ['A', 'B', 'C', 'D']:
            return False
        option_idx = ord(gold_answer) - ord('A')
        choices_list = list(question_data['choices'].values())
        if option_idx >= len(choices_list):
            return False
        answer_text = choices_list[option_idx].lower()

        # Check if answer text appears in context
        return answer_text in context_lower
    else:
        # Open-ended task - check if any gold answer appears in context
        if isinstance(gold_answer, list):
            return any(ans.lower() in context_lower for ans in gold_answer)
        else:
            return str(gold_answer).lower() in context_lower


def classify_sample(question_id, no_ctx_correct, context_text, gold_answer, question_data, is_mc=True):
    """
    Classify each sample into one of three buckets.

    - Corrective: Model wrong without context, context has answer
    - Neutral: Model correct without context, context has answer
    - Misleading: Model wrong without context, context lacks answer
    - Other: Model correct without context but context contradicts (discard)
    """
    context_has_answer = check_context_support(context_text, gold_answer, question_data, is_mc)

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


def evaluate_open(pred, gold_answers):
    """Evaluate open-ended prediction"""
    if pred is None or pred == '':
        return False
    pred_lower = str(pred).lower().strip()
    return any(ans.lower() in pred_lower or pred_lower in ans.lower()
               for ans in gold_answers)


def run_threeway_bucket_analysis(args):
    """Main three-way bucket analysis logic"""

    # Determine dataset type
    is_mc = args.dataset in ['infoseek', 'viquae']

    # Load results
    print(f"Loading results for {args.dataset}...")
    if is_mc:
        no_ctx_results = load_results_jsonl(args.no_context_file)
        regular_mrag_results = load_results_jsonl(args.regular_mrag_file)
        alfar_results = load_results_jsonl(args.alfar_file)
        ground_truth = load_ground_truth_mc(args.question_file)
        contexts = load_contexts_mc(args.question_file, args.dataset)
    else:
        no_ctx_results = load_results_csv(args.no_context_file)
        regular_mrag_results = load_results_csv(args.regular_mrag_file)
        alfar_results = load_results_csv(args.alfar_file)
        ground_truth = load_ground_truth_okvqa(args.question_file)
        contexts = load_contexts_okvqa(args.dataset)

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

        if is_mc:
            no_ctx_correct = evaluate_mc(no_ctx_pred, gold_answer)
            regular_mrag_correct = evaluate_mc(regular_mrag_pred, gold_answer)
            alfar_correct = evaluate_mc(alfar_pred, gold_answer)
        else:
            no_ctx_correct = evaluate_open(no_ctx_pred, gold_answer)
            regular_mrag_correct = evaluate_open(regular_mrag_pred, gold_answer)
            alfar_correct = evaluate_open(alfar_pred, gold_answer)

        # Get context
        context = contexts.get(qid, contexts.get(str(qid), ''))

        # Classify into bucket based on NO-CONTEXT performance
        bucket = classify_sample(qid, no_ctx_correct, context, gold_answer, gt_data, is_mc)

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
            "context_preview": context[:200] + "..." if len(context) > 200 else context
        }

        buckets[bucket].append(sample_info)
        bucket_assignments[qid] = bucket

    # Compute statistics
    print("\n" + "="*80)
    print("THREE-WAY BUCKET ANALYSIS RESULTS")
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
    with open(output_dir / f'{args.dataset}_bucket_assignments.json', 'w') as f:
        json.dump(bucket_assignments, f, indent=2)

    # Save bucket samples
    with open(output_dir / f'{args.dataset}_bucket_samples.json', 'w') as f:
        json.dump(buckets, f, indent=2)

    # Save summary statistics
    with open(output_dir / f'{args.dataset}_bucket_stats.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - {args.dataset}_bucket_assignments.json")
    print(f"  - {args.dataset}_bucket_samples.json")
    print(f"  - {args.dataset}_bucket_stats.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Three-way bucket analysis for motivation experiment")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["infoseek", "viquae", "aokvqa", "okvqa"],
                       help="Dataset name")
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
