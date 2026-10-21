#!/usr/bin/env python3
"""Build a per-(recording, keyword) reliability dataset for C2.

For each ASR system with logged confidence, each recording, and each rare keyword that occurs in the
reference, emit one row: did the system DROP that keyword (label=1, an unreliable keyword query) or
preserve it (0), plus CHEAP signals computable at query time WITHOUT the gold transcript
(ASR confidence avg_logprob, no_speech_prob, keyword rarity df/selectivity, keyword length, hyp length)
and, for comparison, the gold-requiring WER/CER of the recording. reliability_eval.py then tests whether
the cheap gold-free signals predict per-query correctness better than WER.

Usage (.venvASR312, CPU):
  python scripts/reliability_data.py --manifest data/manifests/afrispeech.csv \
    --workload workload/afrispeech_queries.jsonl --asr-dir results/afrispeech/asr \
    --systems tiny.en,small.en,large-v3 --corpus afrispeech --out results/reliability/afrispeech.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import read_manifest, read_jsonl, tokens, wer_sdi, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--systems", required=True, help="comma list; must have avg_logprob (whisper tiers)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    utts = read_manifest(args.manifest)
    ref = {}
    for u in utts:
        ref.setdefault(u.recording_id, []).append(u.ref_text)
    ref = {rid: " ".join(t) for rid, t in ref.items()}

    kws = [(q["keyword"], q.get("df"), q.get("selectivity"))
           for q in read_jsonl(args.workload) if q.get("family") == "lexical"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corpus", "system", "recording_id", "keyword", "dropped",
                    "avg_logprob", "no_speech_prob", "df", "selectivity", "kw_len", "hyp_wc", "wer", "cer"])
        for s in args.systems.split(","):
            hp = Path(args.asr_dir) / s / "hyps.jsonl"
            hyps = {r["recording_id"]: r for r in read_jsonl(hp)}
            for rid, rt in ref.items():
                h = hyps.get(rid)
                if not h:
                    continue
                htext = h.get("hyp_text", "")
                conf = h.get("avg_logprob")
                nsp = h.get("no_speech_prob")
                if conf is None:
                    continue  # system lacks confidence -> skip (only whisper tiers qualify)
                sub, dele, ins, nn = wer_sdi(rt, htext)
                wer = (sub + dele + ins) / nn if nn else 0.0
                cer = (sub + dele) / nn if nn else 0.0
                rtok, htok = set(tokens(rt)), set(tokens(htext))
                hyp_wc = len(tokens(htext))
                for kw, df, sel in kws:
                    if kw not in rtok:
                        continue  # only recordings that actually contain the keyword
                    dropped = 0 if kw in htok else 1
                    w.writerow([args.corpus, s, rid, kw, dropped, conf, nsp, df, sel,
                                len(kw), hyp_wc, round(wer, 4), round(cer, 4)])
                    n += 1
    log(f"wrote {n} (recording,keyword) rows -> {args.out}")


if __name__ == "__main__":
    main()
