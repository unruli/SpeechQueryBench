#!/usr/bin/env python3
"""AMI-100 pairwise bootstrap for lexical F1 — same as pairwise_significance.py but on ami100."""
from __future__ import annotations
import csv, os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float('nan')
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    if pooled == 0: return float('nan')
    return abs(float(np.mean(a) - np.mean(b))) / pooled

rdir = Path(os.environ.get('ECHODB_RESULTS', 'results'))
outdir = rdir / 'stats'
outdir.mkdir(parents=True, exist_ok=True)

# AMI-100
rows100 = list(csv.DictReader(open(rdir / 'ami100' / 'per_query.csv')))
lex100 = defaultdict(dict)
for r in rows100:
    if r.get('family','') == 'lexical' and r.get('lex_f1',''):
        lex100[r['system']][r['qid']] = float(r['lex_f1'])

# AMI-34
rows34 = list(csv.DictReader(open(rdir / 'ami' / 'per_query.csv')))
lex34 = defaultdict(dict)
for r in rows34:
    if r.get('family','') == 'lexical' and r.get('lex_f1',''):
        lex34[r['system']][r['qid']] = float(r['lex_f1'])

B = 5000
for label, lex in [("ami34", lex34), ("ami100", lex100)]:
    systems = sorted(lex.keys())
    qids = sorted(set.intersection(*[set(lex[s]) for s in systems]))
    n = len(qids)
    print(f"\n=== {label}: {len(systems)} systems, {n} queries ===")
    A = {s: np.array([lex[s][q] for q in qids]) for s in systems}
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(B, n))
    boot = {s: A[s][idx].mean(axis=1) for s in systems}
    means = {s: float(np.mean(A[s])) for s in systems}

    # Focus on key pairs
    key_pairs = []
    # Find best and worst system
    ranked = sorted(systems, key=lambda s: means[s], reverse=True)
    print(f"  Ranked: {[(s, f'{means[s]:.4f}') for s in ranked[:5]]}")
    # Key comparisons
    for s1, s2 in [("large-v3", "q3asr"), ("canaryq", "q3asr"), (ranked[0], ranked[-1])]:
        if s1 in systems and s2 in systems:
            diff = boot[s1] - boot[s2]
            p = 2 * min(float(np.mean(diff > 0)), float(np.mean(diff < 0)))
            p = min(p, 1.0)
            d = cohens_d(A[s1], A[s2])
            delta = means[s1] - means[s2]
            print(f"  {s1:20s} vs {s2:20s}: delta={delta:+.4f}, p={p:.4f}, d={d:.2f}")
            key_pairs.append((s1, s2, delta, p, d, n))

    with open(outdir / f'pairwise_{label}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system1','system2','delta_f1','p_value','cohens_d','n'])
        for s1, s2, delta, p, d, nn in key_pairs:
            w.writerow([s1, s2, f"{delta:+.4f}", f"{p:.4f}", f"{d:.2f}", nn])
    print(f"  Written: {outdir / f'pairwise_{label}.csv'}")
print("\nDone.")
