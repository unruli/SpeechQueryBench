#!/usr/bin/env python3
"""Build a manifest from AfriSpeech-Dialogue (intronhealth/afrispeech-dialog) -- accented clinical dialogue.

Each recording is a ~2.5 min simulated doctor-patient conversation in African-accented English (single
channel). metadata_asr_clean.csv columns: file_name,transcript,domain,duration,age_group,accent,country.
We keep accent/country/domain in a sidecar for accent-stratified analysis. One manifest row per dialogue
(recording_id = file basename).

Usage:
    python scripts/build_manifest_afrispeech.py \
        --root ${SQB_MODELS} \
        --meta metadata_asr_clean.csv --out "$ECHODB_MANIFESTS/afrispeech.csv"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import Utt, write_manifest, write_jsonl, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="afrispeech_dialog dir (contains data/ + metadata csv)")
    ap.add_argument("--meta", default="metadata_asr_clean.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-out", default="", help="optional sidecar jsonl with accent/country/domain")
    args = ap.parse_args()

    root = Path(args.root)
    utts, sidecar = [], []
    with open(root / args.meta, newline="") as f:
        for r in csv.DictReader(f):
            fn = r["file_name"]                      # e.g. data/<uuid>_..._..wav
            uid = Path(fn).stem
            wav = root / fn
            if not wav.exists():
                log(f"WARN missing audio {fn}")
                continue
            utts.append(Utt(utt_id=uid, recording_id=uid, dataset="afrispeech_dialog",
                            speaker_id="multi", audio_path=str(wav),
                            ref_text=r.get("transcript", ""),
                            duration=float(r.get("duration", 0) or 0)))
            sidecar.append({"recording_id": uid, "accent": r.get("accent", ""),
                            "country": r.get("country", ""), "domain": r.get("domain", "")})

    n = write_manifest(args.out, utts)
    accents = {}
    for s in sidecar:
        accents[s["accent"]] = accents.get(s["accent"], 0) + 1
    if args.meta_out:
        write_jsonl(args.meta_out, sidecar)
    log(f"wrote {n} dialogues -> {args.out}; accents: {dict(sorted(accents.items(), key=lambda x:-x[1]))}")


if __name__ == "__main__":
    main()
