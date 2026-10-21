#!/usr/bin/env python3
"""Score interaction/temporal queries vs RTTM gold.

Query family (per meeting): n_speakers, n_turns, n_changes, has_overlap, overlap_frac, dominant_share.
Architectures:
  * transcript-only (any of the ASR / audio-LLM systems): a plain transcript has NO speaker structure,
    so it CANNOT answer these queries at all -- reported as N/A (structural incapability). This is the
    point: the dominant transcribe-then-search paradigm cannot serve an entire query family.
  * diarization (pyannote) and any other --pred: scored against gold.

Metrics: n_speakers/n_turns/n_changes -> MAE (and exact-match acc for n_speakers);
has_overlap -> accuracy; overlap_frac/dominant_share -> MAE.

Usage:
    python scripts/interaction_eval.py --gold results/ami/interaction/gold.jsonl \
        --pred pyannote=results/ami/interaction/pyannote.jsonl [--pred speech-llm=...]
"""
from __future__ import annotations

import argparse
from statistics import mean

from common import read_jsonl, log


def load(path):
    return {r["meeting"]: r for r in read_jsonl(path)}


def score(gold, pred, recs):
    def mae(k):
        vs = [abs(gold[m][k] - pred[m][k]) for m in recs if k in pred.get(m, {}) and pred[m][k] is not None]
        return mean(vs) if vs else float("nan")

    n_exact = [1.0 if pred.get(m, {}).get("n_speakers") == gold[m]["n_speakers"] else 0.0 for m in recs]
    ov_acc = [1.0 if pred.get(m, {}).get("has_overlap") == gold[m]["has_overlap"] else 0.0 for m in recs]
    return {
        "spk_MAE": mae("n_speakers"),
        "spk_acc": mean(n_exact) if n_exact else float("nan"),
        "turns_MAE": mae("n_turns"),
        "changes_MAE": mae("n_changes"),
        "overlap_acc": mean(ov_acc) if ov_acc else float("nan"),
        "ovfrac_MAE": mae("overlap_frac"),
        "domshare_MAE": mae("dominant_share"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", action="append", default=[], help="name=path (repeatable)")
    args = ap.parse_args()

    gold = load(args.gold)
    recs = sorted(gold)
    log(f"gold: {len(recs)} meetings; median speakers="
        f"{sorted(gold[m]['n_speakers'] for m in recs)[len(recs)//2]}, all have overlap="
        f"{all(gold[m]['has_overlap'] for m in recs)}")

    print(f"\n=== INTERACTION QUERIES (per-meeting, n={len(recs)}) ===")
    print(f"{'architecture':<16} {'spkMAE':>7} {'spkAcc':>7} {'turnMAE':>8} {'ovAcc':>6} {'ovFrMAE':>8} {'domMAE':>7}")
    print(f"{'transcript-only':<16} {'N/A':>7} {'N/A':>7} {'N/A':>8} {'N/A':>6} {'N/A':>8} {'N/A':>7}"
          "   <- no speaker structure in a transcript")
    for spec in args.pred:
        name, path = spec.split("=", 1)
        s = score(gold, load(path), recs)
        print(f"{name:<16} {s['spk_MAE']:>7.2f} {s['spk_acc']:>7.2f} {s['turns_MAE']:>8.1f} "
              f"{s['overlap_acc']:>6.2f} {s['ovfrac_MAE']:>8.3f} {s['domshare_MAE']:>7.3f}")
    print("\n(transcript-based architectures cannot answer any of these; diarization/audio-native can.)\n")


if __name__ == "__main__":
    main()
