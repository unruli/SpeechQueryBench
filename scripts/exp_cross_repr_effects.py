#!/usr/bin/env python3
"""Compute Cohen's d for cross-representation comparisons (paralinguistic + acoustic-context)."""
from __future__ import annotations
import csv, json, os
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

# --- Paralinguistic (CREMA-D): audio-native vs transcript ---
print("=== Paralinguistic (CREMA-D) ===")
methods = {}
for fname in ['qwen.jsonl', 'clap.jsonl', 'egemaps.jsonl', 'prosody.jsonl', 'textllm.jsonl']:
    fpath = rdir / 'cremad' / fname
    if not fpath.exists(): continue
    rows = [json.loads(l) for l in open(fpath)]
    per_item = [1.0 if r['pred'] == r['gold'] else 0.0 for r in rows]
    acc = np.mean(per_item)
    methods[fname.replace('.jsonl','')] = {'acc': acc, 'per_item': np.array(per_item), 'n': len(per_item)}
    print(f"  {fname:20s}: acc={acc:.3f}, n={len(per_item)}")

# Key comparison: best audio-native vs transcript
audio_methods = {k: v for k, v in methods.items() if k != 'textllm'}
best_audio = max(audio_methods, key=lambda k: audio_methods[k]['acc'])
if 'textllm' in methods:
    a = methods[best_audio]['per_item']
    b = methods['textllm']['per_item']
    # Align by length (take min)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = cohens_d(a, b)
    delta = float(np.mean(a) - np.mean(b))
    print(f"  {best_audio} vs textllm: delta={delta:+.3f}, d={d:.2f}, n={n}")

# --- Acoustic-context: CLAP vs transcript ---
print("\n=== Acoustic-context ===")
ac_methods = {}
for fname in ['clap.jsonl', 'textllm.jsonl']:
    fpath = rdir / 'acoustic' / fname
    if not fpath.exists(): continue
    rows = [json.loads(l) for l in open(fpath)]
    per_item = [1.0 if r.get('has_music_pred') == r.get('has_music_gold') else 0.0 for r in rows]
    acc = np.mean(per_item)
    ac_methods[fname.replace('.jsonl','')] = {'acc': acc, 'per_item': np.array(per_item), 'n': len(per_item)}
    print(f"  {fname:20s}: acc={acc:.3f}, n={len(per_item)}")

if 'clap' in ac_methods and 'textllm' in ac_methods:
    a = ac_methods['clap']['per_item']
    b = ac_methods['textllm']['per_item']
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = cohens_d(a, b)
    delta = float(np.mean(a) - np.mean(b))
    print(f"  clap vs textllm: delta={delta:+.3f}, d={d:.2f}, n={n}")

# Write summary
with open(outdir / 'cross_repr_effects.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['family','method1','method2','acc1','acc2','delta','cohens_d','n'])
    if 'textllm' in methods:
        a_arr = methods[best_audio]['per_item']
        b_arr = methods['textllm']['per_item']
        nn = min(len(a_arr), len(b_arr))
        w.writerow(['paralinguistic', best_audio, 'textllm',
                     f"{methods[best_audio]['acc']:.3f}", f"{methods['textllm']['acc']:.3f}",
                     f"{float(np.mean(a_arr[:nn]) - np.mean(b_arr[:nn])):+.3f}",
                     f"{cohens_d(a_arr[:nn], b_arr[:nn]):.2f}", nn])
    if 'clap' in ac_methods and 'textllm' in ac_methods:
        a_arr = ac_methods['clap']['per_item']
        b_arr = ac_methods['textllm']['per_item']
        nn = min(len(a_arr), len(b_arr))
        w.writerow(['acoustic-ctx', 'clap', 'textllm',
                     f"{ac_methods['clap']['acc']:.3f}", f"{ac_methods['textllm']['acc']:.3f}",
                     f"{float(np.mean(a_arr[:nn]) - np.mean(b_arr[:nn])):+.3f}",
                     f"{cohens_d(a_arr[:nn], b_arr[:nn]):.2f}", nn])
print(f"\nWritten: {outdir / 'cross_repr_effects.csv'}")
print("Done.")
