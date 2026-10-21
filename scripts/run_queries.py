#!/usr/bin/env python3
"""Execute the workload over the reference (gold) and over each ASR system's transcripts.

For each system we reconstruct recording-level text by concatenating that system's utterance
hypotheses, then evaluate every query with the SAME operator code used for gold. Output:
results/queries/<system>.jsonl  (system 'reference' is the gold).

Operators:
  lexical           -> result set = {recording_id : keyword in its tokens}
  aggregate/count   -> |result set|
  aggregate/prop_by_speaker -> {speaker : fraction of that speaker's recordings containing keyword}

Usage:
    python scripts/run_queries.py --manifest M.csv --workload workload/queries.jsonl \
        --asr-dir results/asr --out-dir results/queries
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common import read_manifest, read_jsonl, write_jsonl, tokens, log


def reference_rec_text(utts) -> dict[str, str]:
    buf = defaultdict(list)
    for u in utts:
        buf[u.recording_id].append(u.ref_text)
    return {rid: " ".join(t) for rid, t in buf.items()}


def system_rec_text(hyps_path: Path) -> dict[str, str]:
    buf = defaultdict(list)
    for r in read_jsonl(hyps_path):
        buf[r["recording_id"]].append(r.get("hyp_text", ""))
    return {rid: " ".join(t) for rid, t in buf.items()}


def rec_speaker_map(utts) -> dict[str, str]:
    return {u.recording_id: u.speaker_id for u in utts}


def answer_query(q, rec_tokens, rec_spk, spk_recs):
    kw = q["keyword"]
    hit = {rid for rid, toks in rec_tokens.items() if kw in toks}
    if q["family"] == "lexical":
        return {"result_set": sorted(hit)}
    if q["op"] == "count":
        return {"count": len(hit)}
    if q["op"] == "prop_by_speaker":
        prop = {}
        for spk, recs in spk_recs.items():
            denom = len(recs)
            num = sum(1 for rid in recs if rid in hit)
            prop[spk] = round(num / denom, 6) if denom else 0.0
        return {"prop": prop}
    raise ValueError(q)


def run_system(name, rec_text, workload, rec_spk, spk_recs, out_dir):
    rec_tokens = {rid: set(tokens(t)) for rid, t in rec_text.items()}
    rows = []
    for q in workload:
        ans = answer_query(q, rec_tokens, rec_spk, spk_recs)
        rows.append({"qid": q["qid"], "family": q["family"], "op": q.get("op"), **ans})
    out = Path(out_dir) / f"{name}.jsonl"
    write_jsonl(out, rows)
    log(f"  system={name}: {len(rows)} answers -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    utts = read_manifest(args.manifest)
    workload = list(read_jsonl(args.workload))
    rec_spk = rec_speaker_map(utts)
    spk_recs = defaultdict(list)
    for rid, spk in rec_spk.items():
        spk_recs[spk].append(rid)

    # gold
    run_system("reference", reference_rec_text(utts), workload, rec_spk, spk_recs, args.out_dir)
    # each ASR system
    for hyps in sorted(Path(args.asr_dir).glob("*/hyps.jsonl")):
        name = hyps.parent.name
        run_system(name, system_rec_text(hyps), workload, rec_spk, spk_recs, args.out_dir)


if __name__ == "__main__":
    main()
