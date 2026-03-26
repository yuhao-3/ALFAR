#!/bin/bash
# Check progress of E-VQA evaluations

echo "================================"
echo "E-VQA Evaluation Progress"
echo "================================"
echo ""

for method in alfar tcvm; do
    for seed in 0 1 2 3 4; do
        file="logs/${method}_evqa_seed${seed}_metrics.txt"
        if [ -f "$file" ]; then
            # Check if evaluation is complete (has "Accuracy" in output)
            if grep -q "Accuracy" "$file" 2>/dev/null; then
                acc=$(grep "Accuracy" "$file" | head -1)
                echo "✓ ${method} seed ${seed}: COMPLETE - $acc"
            else
                # Check progress
                progress=$(grep -oP '\d+%' "$file" 2>/dev/null | tail -1)
                items=$(grep -oP '\d+/700' "$file" 2>/dev/null | tail -1)
                if [ -n "$items" ]; then
                    echo "⧗ ${method} seed ${seed}: RUNNING - $items ($progress)"
                else
                    lines=$(wc -l < "$file" 2>/dev/null)
                    echo "⧗ ${method} seed ${seed}: STARTING - $lines lines"
                fi
            fi
        else
            echo "✗ ${method} seed ${seed}: NOT STARTED"
        fi
    done
done

echo ""
echo "================================"
echo "Summary"
echo "================================"

complete=$(grep -l "Accuracy" logs/*evqa_seed*_metrics.txt 2>/dev/null | wc -l)
total=10

echo "Complete: $complete/$total"
echo "Running: $((total - complete))"

if [ "$complete" -eq "$total" ]; then
    echo ""
    echo "All E-VQA evaluations complete!"
    echo "Run: python scripts/calculate_all_averages.py"
fi
