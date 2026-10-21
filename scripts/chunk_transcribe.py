#!/usr/bin/env python3
"""Produce chunk-aligned ASR text for retrieval-gating (faster-whisper large-v3 with timestamps).

For each recording, transcribe with segment timestamps and bin segments into fixed windows, so each
30s chunk gets its ASR text. Output: <out-dir>/<recording_id>.jsonl, rows {chunk_idx,start,end,text}.
Used by sem_hybrid.py to pick which audio chunks to hand to the speech LLM (retrieval-gate).

Usage (project .venv, GPU):
    python scripts/chunk_transcribe.py --manifest AMI.csv --model large-v3 \
        --out-dir results/ami/semantic/chunks --chunk-sec 30 --limit 12
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

from common import read_manifest, write_jsonl, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-sec", type=float, default=30.0)
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    log(f"loading {args.model} ...")
    t0 = time.time()
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)
    log(f"loaded in {time.time()-t0:.1f}s")

    # unique recordings, in manifest order
    utts = read_manifest(args.manifest)
    rec_audio, order = {}, []
    for u in utts:
        if u.recording_id not in rec_audio:
            rec_audio[u.recording_id] = u.audio_path
            order.append(u.recording_id)
    if args.limit:
        order = order[:args.limit]

    for i, rid in enumerate(order, 1):
        segs, _ = model.transcribe(rec_audio[rid], language="en", beam_size=5)
        bins = defaultdict(list)
        for s in segs:
            bins[int(s.start // args.chunk_sec)].append(s.text.strip())
        rows = [{"chunk_idx": k, "start": k * args.chunk_sec, "end": (k + 1) * args.chunk_sec,
                 "text": " ".join(bins[k])} for k in sorted(bins)]
        write_jsonl(Path(args.out_dir) / f"{rid}.jsonl", rows)
        log(f"  {i}/{len(order)} {rid}: {len(rows)} chunks")


if __name__ == "__main__":
    main()
