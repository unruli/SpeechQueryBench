#!/usr/bin/env python3
"""Sweep semantic binarization threshold for score-based methods (embed, CLAP) and vote-ratio
for speechLLM QA. Reports F1 at each threshold against gold."""
from __future__ import annotations
import csv, json, os
from pathlib import Path
import numpy as np

rdir = Path(os.environ.get('ECHODB_RESULTS', 'results'))
outdir = rdir / 'stats'
outdir.mkdir(parents=True, exist_ok=True)
sem = rdir / 'ami' / 'semantic'

# Load gold
gold = {}
for r in (json.loads(l) for l in open(sem / 'gold.jsonl')):
    gold[r['recording_id']] = set(r.get('concepts', []))
recs = sorted(gold.keys())
all_concepts = set()
for v in gold.values(): all_concepts |= v

def f1_at_threshold(scores_by_rec, threshold):
    tp = fp = fn = 0
    for rid in recs:
        if rid not in scores_by_rec: continue
        pred = {int(c) for c, s in scores_by_rec[rid].items() if s >= threshold}
        g = gold[rid]
        tp += len(pred & g)
        fp += len(pred - g)
        fn += len(g - pred)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f = 2*p*r/(p+r) if (p+r) else 0
    return p, r, f

# Load score-based methods
methods = {}
for name, fname in [('text-embedding', 'embed_largev3.jsonl'), ('CLAP', 'clap.jsonl')]:
    fpath = sem / fname
    if not fpath.exists(): continue
    scores = {}
    for r in (json.loads(l) for l in open(fpath)):
        scores[r['recording_id']] = r.get('scores', {})
    methods[name] = scores

# SpeechLLM QA: normalize votes to [0,1]
fpath = sem / 'speechllm_qa.jsonl'
if fpath.exists():
    scores = {}
    for r in (json.loads(l) for l in open(fpath)):
        nc = r.get('n_chunks', 1)
        scores[r['recording_id']] = {c: v/nc for c, v in r.get('votes', {}).items()}
    methods['speech-LLM QA'] = scores

# ASR+LLM is binary (no scores), compute single F1
asrllm_path = sem / 'asrllm_largev3.jsonl'
asrllm_f1 = None
if asrllm_path.exists():
    tp = fp = fn = 0
    for r in (json.loads(l) for l in open(asrllm_path)):
        rid = r['recording_id']
        if rid not in gold: continue
        pred = set(r.get('concepts', []))
        g = gold[rid]
        tp += len(pred & g); fp += len(pred - g); fn += len(g - pred)
    p = tp/(tp+fp) if (tp+fp) else 0
    r_ = tp/(tp+fn) if (tp+fn) else 0
    asrllm_f1 = 2*p*r_/(p+r_) if (p+r_) else 0
    print(f"ASR+text-LLM (binary): F1={asrllm_f1:.3f}")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"\nThreshold sweep ({len(recs)} recordings, {len(all_concepts)} concepts):")
results = []
for name, scores in methods.items():
    print(f"\n  {name}:")
    for t in thresholds:
        p, r, f = f1_at_threshold(scores, t)
        print(f"    t={t:.1f}: P={p:.3f} R={r:.3f} F1={f:.3f}")
        results.append({'method': name, 'threshold': t, 'precision': f'{p:.3f}',
                       'recall': f'{r:.3f}', 'f1': f'{f:.3f}'})

with open(outdir / 'semantic_threshold_sweep.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['method','threshold','precision','recall','f1'])
    w.writeheader()
    w.writerows(results)
    if asrllm_f1 is not None:
        w.writerow({'method': 'ASR+text-LLM', 'threshold': 'binary', 'precision': '', 'recall': '', 'f1': f'{asrllm_f1:.3f}'})
print(f"\nWritten: {outdir / 'semantic_threshold_sweep.csv'}")
print("Done.")
