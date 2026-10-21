#!/usr/bin/env python3
"""Build a recording-level manifest from raw LibriSpeech.

LibriSpeech layout:  <root>/<split>/<spk>/<chap>/<spk>-<chap>-<utt>.flac
                     <root>/<split>/<spk>/<chap>/<spk>-<chap>.trans.txt   (lines: "<utt_id> TEXT")

We keep utterance-level rows but tag each with recording_id=<spk>-<chap>, so downstream query
operators can aggregate at the "recording" (chapter) granularity. Chapters are natural documents:
they map to a book section, giving coherent topical content for keyword/aggregate queries.

Usage:
    python scripts/build_manifest_librispeech.py \
        --root "$LIBRISPEECH_ROOT" --split "$LIBRISPEECH_SPLIT" \
        --out  "$ECHODB_MANIFESTS/librispeech_test_clean.csv"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import Utt, write_manifest, log

try:
    import soundfile as sf  # optional; only for duration
except Exception:  # pragma: no cover
    sf = None


def duration_of(flac: Path) -> float:
    if sf is None:
        return 0.0
    try:
        info = sf.info(str(flac))
        return round(info.frames / float(info.samplerate), 3)
    except Exception:
        return 0.0


def build(root: Path, split: str):
    split_dir = root / split
    if not split_dir.is_dir():
        raise SystemExit(f"split dir not found: {split_dir}")
    for trans in sorted(split_dir.glob("*/*/*.trans.txt")):
        chap_dir = trans.parent
        rec_id = trans.stem.replace(".trans", "")  # <spk>-<chap>
        spk = rec_id.split("-")[0]
        with open(trans) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, text = line.split(" ", 1)
                flac = chap_dir / f"{utt_id}.flac"
                if not flac.exists():
                    log(f"WARN missing audio for {utt_id}")
                    continue
                yield Utt(
                    utt_id=utt_id,
                    recording_id=rec_id,
                    dataset="librispeech_" + split.replace("-", "_"),
                    speaker_id=spk,
                    audio_path=str(flac),
                    ref_text=text,
                    duration=duration_of(flac),
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    utts = list(build(Path(args.root), args.split))
    n = write_manifest(args.out, utts)
    recs = len({u.recording_id for u in utts})
    spks = len({u.speaker_id for u in utts})
    log(f"wrote {n} utterances across {recs} recordings / {spks} speakers -> {args.out}")


if __name__ == "__main__":
    main()
