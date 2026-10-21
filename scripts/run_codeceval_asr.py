#!/usr/bin/env python3
"""Transcribe our manifests with the CodecEval-Africa SOTA ASR models (Parakeet-TDT, Canary-Qwen,
Qwen3-ASR), emitting hyps.jsonl in the same schema as run_asr.py so they drop straight into the
query/WER/CER/cost pipeline as new systems. Reuses the proven ASREvaluator (model loading + 30 s
sliding-window chunking) from CodecEval-Africa so we don't re-implement model handling.

Checkpointing: appends per utterance and resumes from an existing hyps.jsonl (skips done utt_ids), so a
timeout/thrash doesn't lose progress on the long corpora.

Usage (.venvASR, GPU; PYTHONPATH includes CodecEval code):
    python scripts/run_codeceval_asr.py --manifest data/manifests/afrispeech.csv \
        --model nvidia/parakeet-tdt-0.6b-v2 --system parakeet \
        --out-dir results/afrispeech/asr --chunk-sec 30 [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from common import read_manifest, log

CODECEVAL_CODE = "${SQB_MODELS}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True, help="HF id, e.g. nvidia/parakeet-tdt-0.6b-v2")
    ap.add_argument("--system", required=True, help="tag for results/<...>/asr/<system>/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-sec", type=float, default=30.0)
    ap.add_argument("--language", default="en")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    utts = read_manifest(args.manifest)
    if args.limit:
        utts = utts[:args.limit]

    od = Path(args.out_dir) / args.system
    od.mkdir(parents=True, exist_ok=True)
    hp = od / "hyps.jsonl"
    done = set()
    if hp.exists():
        for line in open(hp):
            try:
                done.add(json.loads(line)["utt_id"])
            except Exception:  # noqa: BLE001
                pass
        log(f"resuming: {len(done)} already done")
    todo = [u for u in utts if u.utt_id not in done]
    if not todo:
        log(f"nothing to do ({len(done)}/{len(utts)} present) -> {hp}")
        return

    # Load asr_evaluator.py DIRECTLY (bypass the audio_quality_assessment package __init__, which pulls
    # in dnsmos/quality-scorer deps we don't need for transcription).
    import importlib.util
    sys.path.insert(0, CODECEVAL_CODE)
    _p = f"{CODECEVAL_CODE}/audio_quality_assessment/asr_evaluator.py"
    _spec = importlib.util.spec_from_file_location("asr_evaluator", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ASREvaluator = _mod.ASREvaluator
    log(f"loading {args.model} (chunk={args.chunk_sec}s) ...")
    t0 = time.time()
    ev = ASREvaluator(model_name=args.model, chunk_seconds=args.chunk_sec, language=args.language)
    log(f"loaded in {time.time()-t0:.0f}s; transcribing {len(todo)} utts")

    t0 = time.time()
    with open(hp, "a") as f:
        for i, u in enumerate(todo, 1):
            try:
                r = ev._transcribe_audio(u.audio_path)
                hyp = ((r or {}).get("text", "") or "").strip()
            except Exception as e:  # noqa: BLE001
                log(f"WARN {u.utt_id}: {e}")
                hyp = ""
            f.write(json.dumps({"utt_id": u.utt_id, "recording_id": u.recording_id,
                                "hyp_text": hyp}) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                log(f"  {i}/{len(todo)} ({time.time()-t0:.0f}s)")
    log(f"wrote {len(todo)} new hyps -> {hp}")


if __name__ == "__main__":
    main()
