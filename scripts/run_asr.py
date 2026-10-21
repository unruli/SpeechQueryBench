#!/usr/bin/env python3
"""Transcribe a manifest with ONE faster-whisper model (load once, stream all utts).

Run once per ASR tier (see runners/sbatch_asr.sh). Output: results/asr/<model>/hyps.jsonl
with rows {utt_id, hyp_text, avg_logprob, no_speech_prob} — the extra fields feed RQ6
(cheap uncertainty signals) later.

Usage:
    python scripts/run_asr.py --manifest M.csv --model large-v3 --out-dir results/asr \
        --compute-type float16 [--limit N]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import read_manifest, write_jsonl, log


def transcribe_all(manifest, model_name, compute_type, limit):
    from faster_whisper import WhisperModel  # imported here so --help works without deps

    device = "cuda"
    log(f"loading {model_name} ({compute_type}) on {device} ...")
    t0 = time.time()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    log(f"loaded in {time.time()-t0:.1f}s")

    utts = read_manifest(manifest)
    if limit:
        utts = utts[:limit]
    n = len(utts)
    for i, u in enumerate(utts, 1):
        segs, _info = model.transcribe(u.audio_path, language="en", beam_size=5)
        parts, lp, nsp, k = [], 0.0, 0.0, 0
        for s in segs:
            parts.append(s.text)
            lp += getattr(s, "avg_logprob", 0.0) or 0.0
            nsp += getattr(s, "no_speech_prob", 0.0) or 0.0
            k += 1
        yield {
            "utt_id": u.utt_id,
            "recording_id": u.recording_id,
            "hyp_text": " ".join(p.strip() for p in parts).strip(),
            "avg_logprob": round(lp / k, 4) if k else None,
            "no_speech_prob": round(nsp / k, 4) if k else None,
        }
        if i % 200 == 0 or i == n:
            log(f"  {i}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True, help="faster-whisper id, e.g. tiny.en / small.en / large-v3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir) / args.model.replace("/", "_") / "hyps.jsonl"
    t0 = time.time()
    rows = list(transcribe_all(args.manifest, args.model, args.compute_type, args.limit))
    write_jsonl(out, rows)
    log(f"model={args.model}: {len(rows)} hyps in {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
