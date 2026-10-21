#!/usr/bin/env python3
"""Diarization architecture for interaction/temporal queries (pyannote speaker-diarization-3.1).

Transcript-only architectures cannot answer speaker-structure queries at all (no speaker labels); a
diarization pipeline can. This runs pyannote over each meeting and computes the SAME features as the
RTTM gold (interaction_gold.meeting_features), so interaction_eval.py can score it.

Output: preds jsonl {meeting, n_speakers, n_turns, n_changes, has_overlap, overlap_frac, dominant_share}.

Usage (Diarizer/.venv, GPU):
    python scripts/interaction_diarize.py --manifest AMI.csv --out results/ami/interaction/pyannote.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import read_manifest, write_jsonl, log
from interaction_gold import meeting_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from pyannote.audio import Pipeline
    log(f"loading {args.model} ...")
    t0 = time.time()
    pipeline = Pipeline.from_pretrained(args.model)
    pipeline.to(torch.device("cuda"))
    log(f"loaded in {time.time()-t0:.1f}s")

    utts = read_manifest(args.manifest)
    rec_audio, order = {}, []
    for u in utts:
        if u.recording_id not in rec_audio:
            rec_audio[u.recording_id] = u.audio_path
            order.append(u.recording_id)
    if args.limit:
        order = order[:args.limit]

    out = Path(args.out)
    rows = []
    for i, mid in enumerate(order, 1):
        try:
            diar = pipeline(rec_audio[mid])
            segs = [(seg.start, seg.end, spk) for seg, _trk, spk in diar.itertracks(yield_label=True)]
            feat = meeting_features(segs) or {}
        except Exception as e:  # noqa: BLE001
            log(f"WARN {mid}: {e}")
            feat = {}
        rows.append({"meeting": mid, **feat})
        write_jsonl(out, rows)  # checkpoint each meeting (diarization is slow)
        log(f"  {i}/{len(order)} {mid}: spk={feat.get('n_speakers')} turns={feat.get('n_turns')} "
            f"overlap={feat.get('has_overlap')}")

    log(f"wrote {len(rows)} meetings -> {out}")


if __name__ == "__main__":
    main()
