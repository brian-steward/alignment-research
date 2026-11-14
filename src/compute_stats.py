#!/usr/bin/env python3
"""Compute statistics for scored batches."""
import json
import math

def compute_stats(filepath):
    scores = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                scores.append(obj['score'])
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    std = math.sqrt(variance)
    return n, mean, std, min(scores), max(scores)

for batch in ['0433', '0434', '0435']:
    path = f'data/llm_scores/batches/batch_{batch}_scored.jsonl'
    n, mean, std, mn, mx = compute_stats(path)
    print(f'Batch {batch}: n={n}, mean={mean:.4f}, std={std:.4f}, min={mn:.1f}, max={mx:.1f}')
