#!/usr/bin/env python3
"""Pairwise bootstrap significance + Cohen's d for lexical F1."""
from __future__ import annotations
import argparse, csv, os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float('nan')
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    if pooled == 0: return float('nan')
    return abs(float(np.mean(a) - np.mean(b))) / pooled

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results/ami')
    ap.add_argument('--B', type=int, default=5000)
    args = ap.parse_args()
    rdir = Path(args.results_dir)
    outdir = Path(os.environ.get('ECHODB_RESULTS', 'results')) / 'stats'
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(rdir / 'per_query.csv')
    # Build per-system per-query F1 for lexical family
    lex = defaultdict(dict)
    for r in rows:
        if r.get('family','') == 'lexical' and r.get('lex_f1',''):
            lex[r['system']][r['qid']] = float(r['lex_f1'])

    systems = sorted(lex.keys())
    if not systems:
        print("ERROR: no lexical data found"); return
    qids = sorted(set.intersection(*[set(lex[s]) for s in systems]))
    n = len(qids)
    print(f"Lexical: {len(systems)} systems, {n} queries")
    A = {s: np.array([lex[s][q] for q in qids]) for s in systems}
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(args.B, n))
    boot = {s: A[s][idx].mean(axis=1) for s in systems}

    with open(outdir / 'pairwise_lexical.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system1','system2','mean1','mean2','delta_f1','p_value','cohens_d','n','significant'])
        pairs = list(combinations(systems, 2))
        pvals = {}
        for s1, s2 in pairs:
            diff = boot[s1] - boot[s2]
            p = 2 * min(float(np.mean(diff > 0)), float(np.mean(diff < 0)))
            p = min(p, 1.0)
            pvals[(s1,s2)] = p

        # Holm-Bonferroni
        sorted_p = sorted(pvals.items(), key=lambda x: x[1])
        m = len(sorted_p)
        sig = {}
        for i, (pair, p) in enumerate(sorted_p):
            sig[pair] = p <= 0.05 / (m - i)

        for s1, s2 in pairs:
            d = cohens_d(A[s1], A[s2])
            delta = float(np.mean(A[s1]) - np.mean(A[s2]))
            w.writerow([s1, s2, f"{np.mean(A[s1]):.4f}", f"{np.mean(A[s2]):.4f}",
                       f"{delta:+.4f}", f"{pvals[(s1,s2)]:.4f}", f"{d:.2f}", n, sig[(s1,s2)]])

    print(f"Written: {outdir / 'pairwise_lexical.csv'}")

    # Summary
    with open(outdir / 'pairwise_summary.txt', 'w') as f:
        f.write(f"=== Pairwise Bootstrap (Lexical F1, B={args.B}, Holm-Bonferroni) ===\n\n")
        f.write(f"Systems: {', '.join(systems)}\n")
        f.write(f"Queries: {n}\n\n")
        for (s1,s2), p in sorted(pvals.items(), key=lambda x: x[1]):
            star = ' *' if sig[(s1,s2)] else ''
            d = cohens_d(A[s1], A[s2])
            delta = float(np.mean(A[s1]) - np.mean(A[s2]))
            f.write(f"  {s1:20s} vs {s2:20s}: delta={delta:+.4f}, p={p:.4f}, d={d:.2f}{star}\n")
        f.write(f"\n* significant at alpha=0.05 after Holm-Bonferroni ({m} comparisons)\n")
    print(f"Written: {outdir / 'pairwise_summary.txt'}")

    # Effect sizes table
    with open(outdir / 'effect_sizes.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['family','system1','system2','delta','cohens_d','n'])
        for s1, s2 in pairs:
            d = cohens_d(A[s1], A[s2])
            delta = float(np.mean(A[s1]) - np.mean(A[s2]))
            w.writerow(['lexical', s1, s2, f"{delta:+.4f}", f"{d:.2f}", n])
    print(f"Written: {outdir / 'effect_sizes.csv'}")
    print("Done.")

if __name__ == '__main__':
    main()
