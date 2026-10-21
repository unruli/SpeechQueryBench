#!/usr/bin/env python3
"""Reliability-aware routing pilot: a learned gate escalates cheap->expensive on predicted-hard recordings.

Operationalizes C1 (cost) + C2 (reliability). Default to a CHEAP system (Parakeet); for each recording,
a gate trained on gold-free features (SNR, duration, cheap-hyp length + keyword count) predicts the
cheap system's rare-keyword recall and escalates the worst-predicted recordings to an EXPENSIVE system
(Q3ASR). We sweep the escalation budget and compare three policies -- random (null), learned (ours,
leave-one-out), oracle (ceiling) -- on micro rare-keyword recall, and map budget->cost. If learned beats
random and approaches oracle, cheap signals let you route smartly: near the expensive system's utility at
a fraction of its cost.

Usage (.venvASR312, CPU):
    python scripts/router_pilot.py --manifest data/manifests/afrispeech.csv \
        --workload workload/afrispeech_queries.jsonl --asr-dir results/afrispeech/asr \
        --cheap parakeet --expensive q3asr --cheap-rtf 0.0071 --exp-rtf 0.0622 --rate 0.40
"""
from __future__ import annotations

import argparse
import numpy as np

from common import read_manifest, read_jsonl, tokens, log


def snr_db(path):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    w = max(1, int(0.025 * sr))
    frames = [y[i:i + w] for i in range(0, len(y) - w, w)]
    e = np.array([float(np.mean(f ** 2)) + 1e-12 for f in frames])
    noise = np.percentile(e, 10); sig = np.percentile(e, 90)
    return 10.0 * np.log10(sig / noise)


def micro_recall(recs, ref_kw, hit_cheap, hit_exp, escalated):
    tot = sum(len(ref_kw[r]) for r in recs)
    hit = sum(len(hit_exp[r] if r in escalated else hit_cheap[r]) for r in recs)
    return hit / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", required=True, help="semicolon-sep 'manifest,workload,asrdir' triples")
    ap.add_argument("--cheap", required=True); ap.add_argument("--expensive", required=True)
    ap.add_argument("--cheap-rtf", type=float, required=True); ap.add_argument("--exp-rtf", type=float, required=True)
    ap.add_argument("--rate", type=float, default=0.40)
    args = ap.parse_args()
    from pathlib import Path

    recs, ref_kw, hit_c, hit_e, feats, ychr, conf = [], {}, {}, {}, [], [], []
    for spec in args.corpora.split(";"):                          # each: manifest,workload,asr-dir
        man, wl, ad = spec.split(",")
        utts = read_manifest(man)
        ref = {u.recording_id: u.ref_text for u in utts}
        dur = {u.recording_id: float(u.duration or 0) for u in utts}
        audio = {u.recording_id: u.audio_path for u in utts}
        K = {q["keyword"] for q in read_jsonl(wl) if q.get("family") == "lexical" and q.get("rarity") == "rare"}
        hyp, clog = {}, {}
        for sysname in (args.cheap, args.expensive):
            d = {}
            for r in read_jsonl(Path(ad) / sysname / "hyps.jsonl"):
                d.setdefault(r["recording_id"], []).append(r.get("hyp_text", ""))
                if sysname == args.cheap and r.get("avg_logprob") is not None:
                    clog[r["recording_id"]] = r["avg_logprob"]
            hyp[sysname] = {k: " ".join(v) for k, v in d.items()}
        for r, rt in ref.items():
            rk = K & set(tokens(rt))
            if not rk:
                continue
            ch = hyp[args.cheap].get(r, ""); ex = hyp[args.expensive].get(r, "")
            ctok, etok = set(tokens(ch)), set(tokens(ex))
            rid = f"{ad}::{r}"
            recs.append(rid); ref_kw[rid] = rk
            hit_c[rid] = rk & ctok; hit_e[rid] = rk & etok
            ychr.append(len(hit_c[rid]) / len(rk))
            dur[rid] = dur.get(r, 0.0)
            feats.append([snr_db(audio[r]), dur.get(r, 0.0), len(tokens(ch)), len(K & ctok), clog.get(r, 0.0)])
            conf.append(clog.get(r, 0.0))
    dur = {r: feats[i][1] for i, r in enumerate(recs)}
    X = np.array(feats); y = np.array(ychr); conf = np.array(conf); n = len(recs)
    log(f"{n} recordings with rare keywords; cheap micro-recall="
        f"{micro_recall(recs, ref_kw, hit_c, hit_e, set()):.3f} expensive="
        f"{micro_recall(recs, ref_kw, hit_c, hit_e, set(recs)):.3f}")

    # leave-one-out predicted cheap-recall (learned gate)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    pred = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[tr], y[tr])
        pred[i] = m.predict(X[i:i + 1])[0]

    order_learned = np.argsort(pred)          # escalate lowest predicted recall first
    order_conf = np.argsort(conf)             # escalate lowest cheap-confidence first (no training)
    order_oracle = np.argsort(y)              # escalate truly-worst first
    rng = np.random.default_rng(0)
    cheap_cost = args.cheap_rtf * sum(dur[r] for r in recs) * args.rate / 3600.0

    print(f"\n=== Reliability-aware routing ({args.cheap} -> {args.expensive}, n={n}), rare-keyword recall vs budget ===")
    print(f"{'budget':>7}{'random':>9}{'conf':>9}{'learned':>9}{'oracle':>9}{'$/all':>9}")
    for b in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        k = int(round(b * n))
        rr = (np.mean([micro_recall(recs, ref_kw, hit_c, hit_e, {recs[j] for j in rng.choice(n, k, replace=False)})
                       for _ in range(100)]) if 0 < k < n
              else micro_recall(recs, ref_kw, hit_c, hit_e, set() if k == 0 else set(recs)))
        rc = micro_recall(recs, ref_kw, hit_c, hit_e, {recs[j] for j in order_conf[:k]})
        rl = micro_recall(recs, ref_kw, hit_c, hit_e, {recs[j] for j in order_learned[:k]})
        ro = micro_recall(recs, ref_kw, hit_c, hit_e, {recs[j] for j in order_oracle[:k]})
        esc_cost = args.exp_rtf * sum(dur[recs[j]] for j in order_conf[:k]) * args.rate / 3600.0
        print(f"{b:>7.1f}{rr:>9.3f}{rc:>9.3f}{rl:>9.3f}{ro:>9.3f}{cheap_cost+esc_cost:>9.4f}")
    print(f"\nalways-cheap $={cheap_cost:.4f}; always-expensive $="
          f"{args.exp_rtf*sum(dur[r] for r in recs)*args.rate/3600.0:.4f}. "
          f"conf = escalate lowest cheap-confidence (no training); learned = LOO gate.\n")


if __name__ == "__main__":
    main()
