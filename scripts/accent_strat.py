#!/usr/bin/env python3
"""Accent-stratified analysis on AfriSpeech: does the WER<->query decoupling vary by accent?

Uses the accent sidecar (recording -> accent) + each system's hyps. Reports:
  * accent difficulty: per-accent WER averaged over systems (which accents are hardest)
  * per-accent WER for each system (does the best-WER system flip by accent)
  * per-accent keyword-recall error (a per-recording query-utility proxy) vs WER
  * whether CER < WER re-ranks toward query utility within accents

Usage:
    python scripts/accent_strat.py --manifest afrispeech.csv --meta afrispeech_meta.jsonl \
        --workload afrispeech_queries.jsonl --asr-dir results/afrispeech/asr [--min-n 4]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from common import read_manifest, read_jsonl, tokens, wer_sdi, log


def keyword_recall_err(ref, hyp, K):
    rtok, htok = set(tokens(ref)), set(tokens(hyp))
    kref = K & rtok
    if not kref:
        return None
    fn = len(kref - htok)
    return fn / len(kref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--min-n", type=int, default=4)
    args = ap.parse_args()

    utts = read_manifest(args.manifest)
    ref = {u.recording_id: u.ref_text for u in utts}
    accent = {r["recording_id"]: r["accent"] for r in read_jsonl(args.meta)}
    K = {q["keyword"] for q in read_jsonl(args.workload) if q.get("family") == "lexical"}
    acc_n = defaultdict(int)
    for rid in ref:
        acc_n[accent.get(rid, "?")] += 1
    accents = [a for a, n in sorted(acc_n.items(), key=lambda x: -x[1]) if n >= args.min_n]

    systems = sorted(p.parent.name for p in Path(args.asr_dir).glob("*/hyps.jsonl"))
    # per (system, accent): micro WER + CER + mean keyword recall_err
    W = defaultdict(lambda: [0, 0, 0])  # (sys,acc) -> [S+D+I, S+D, N]
    RE = defaultdict(list)
    for s in systems:
        hyp = {r["utt_id"]: r.get("hyp_text", "") for r in read_jsonl(Path(args.asr_dir) / s / "hyps.jsonl")}
        for rid, rt in ref.items():
            a = accent.get(rid, "?")
            sub, dele, ins, n = wer_sdi(rt, hyp.get(rid, ""))
            W[(s, a)][0] += sub + dele + ins; W[(s, a)][1] += sub + dele; W[(s, a)][2] += n
            re = keyword_recall_err(rt, hyp.get(rid, ""), K)
            if re is not None:
                RE[(s, a)].append(re)

    def wer(s, a):
        e, _, n = W[(s, a)]; return e / n if n else float("nan")

    def cer(s, a):
        _, c, n = W[(s, a)]; return c / n if n else float("nan")

    print(f"\n=== ACCENT DIFFICULTY (mean WER over {len(systems)} systems; n>= {args.min_n}) ===")
    for a in sorted(accents, key=lambda a: mean(wer(s, a) for s in systems)):
        print(f"  {a:<10} n={acc_n[a]:<3} meanWER={mean(wer(s,a) for s in systems):.3f}  "
              f"kwRecallErr={mean(v for s in systems for v in RE[(s,a)]) if any(RE[(s,a)] for s in systems) else float('nan'):.3f}")

    print(f"\n=== PER-ACCENT WER by system (does the best-WER system flip?) ===")
    print(f"{'accent':<10} " + " ".join(f"{s[:9]:>9}" for s in systems))
    for a in accents:
        row = " ".join(f"{wer(s,a):>9.3f}" for s in systems)
        best = min(systems, key=lambda s: wer(s, a))
        print(f"{a:<10} {row}   best={best}")

    print(f"\n=== WER vs CER best system per accent (CER should track query utility) ===")
    for a in accents:
        bw = min(systems, key=lambda s: wer(s, a))
        bc = min(systems, key=lambda s: cer(s, a))
        print(f"  {a:<10} best-by-WER={bw:<14} best-by-CER={bc}")


if __name__ == "__main__":
    main()
