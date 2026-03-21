#!/usr/bin/env python3
"""
Evaluate OK-VQA and A-OKVQA results against ground truth annotations
"""
import pandas as pd
from ast import literal_eval
import okvqa_evaluation as evaluation
import numpy as np
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='aokvqa', choices=['okvqa', 'aokvqa'])
    parser.add_argument("--input_file", type=str, default=None)
    args = parser.parse_args()

    # Set default input file if not provided
    if args.input_file is None:
        args.input_file = f'experiments/result/{args.dataset}_alfar_results.csv'

    # Load ground truth annotations
    print("Loading ground truth annotations...")
    if args.dataset == 'aokvqa':
        val_annotations_df = pd.read_csv('data/eval_data/okvqa/a_ok_vqa_val_fixed_annots.csv')
    else:  # okvqa
        val_annotations_df = pd.read_csv('data/eval_data/okvqa/val_annots_fixed.csv')

    val_annotations_df['answers'] = val_annotations_df['answers'].apply(literal_eval)

    # Load model predictions
    print("Loading model predictions...")
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found!")
        print(f"Please run the {args.dataset.upper()} evaluation SLURM job first.")
        sys.exit(1)

    results_df = pd.read_csv(args.input_file)

    # Convert llama_answer to string and handle NaN values
    results_df['llama_answer'] = results_df['llama_answer'].fillna('').astype(str)

    # Merge predictions with ground truth
    print("Merging predictions with ground truth...")
    results_df = pd.merge(val_annotations_df, results_df, on='question_id')

    # Calculate accuracy for each question
    print("Calculating accuracy...")

    # Import sys for proper path handling
    import sys
    sys.path.insert(0, 'evaluation')

    results_df['acc'] = results_df.apply(
        lambda row: evaluation.okvqa_ems(str(row['llama_answer']), row['answers'], row['question_id']),
        axis=1
    )

    # Print overall accuracy
    overall_acc = results_df['acc'].mean()
    print("\n" + "="*50)
    print(f"{args.dataset.upper()} Accuracy: {np.round(overall_acc, 4)} ({np.round(overall_acc*100, 2)}%)")
    print(f"Correct: {results_df['acc'].sum()}/{len(results_df)}")
    print("="*50)

    # Show some example predictions
    print("\nSample predictions:")
    print(results_df[['question_id', 'question', 'llama_answer', 'answers', 'acc']].head(10))

    # Show some incorrect predictions for debugging
    incorrect = results_df[results_df['acc'] == 0]
    if len(incorrect) > 0:
        print(f"\nIncorrect predictions: {len(incorrect)}/{len(results_df)}")
        print("\nSample incorrect predictions:")
        print(incorrect[['question_id', 'question', 'llama_answer', 'answers']].head(5))

if __name__ == "__main__":
    main()