#!/usr/bin/env python3
"""Build a MEETING-level manifest from AMI manual (NXT) word annotations + Mix-Headset audio.

AMI is the "hard" anchor: spontaneous multiparty speech with overlap. Reference words come from the
manual NXT annotations (ami_public_manual .../words/<MEETING>.<SPK>.words.xml):
    <w nite:id=".." starttime="0.37" endtime="0.95">Hmm</w>
    <w ... punc="true">.</w>                       # punctuation -> skipped
Files are per meeting per speaker (SPK in A/B/C/D). Audio: $AMI_AUDIO/<MEETING>/audio/<MEETING>.Mix-Headset.wav.

Emits one manifest row per meeting (recording_id=meeting, ref_text = time-ordered words across ALL
speakers) plus a <MEETING>.words.jsonl sidecar (start,end,word,speaker) for temporal/overlap queries.
Note: AMI-diarization-setup provides only RTTM/VAD (no words); this uses the manual annotations.

Usage:
    python scripts/build_manifest_ami.py --annot "$AMI_ANNOT" --words "$AMI_WORDS" \
        --audio "$AMI_AUDIO" --out "$ECHODB_MANIFESTS/ami.csv" \
        --words-dir-out "$ECHODB_ROOT/data/manifests/ami_words"
"""
from __future__ import annotations

import argparse
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

from common import Utt, write_manifest, write_jsonl, log


def read_meetings(annot: Path, splits=("test", "dev")) -> list[str]:
    ids = []
    for split in splits:
        p = annot / "lists" / f"{split}.meetings.txt"
        if p.exists():
            ids += [x.strip() for x in p.read_text().splitlines() if x.strip()]
    return sorted(set(ids))


def read_words(words_dir: Path, meeting: str):
    """Yield (start, end, word, speaker) for real words (skip punctuation + vocalsounds)."""
    for wf in sorted(glob.glob(str(words_dir / f"{meeting}.*.words.xml"))):
        spk = Path(wf).name.split(".")[1]  # A/B/C/D
        try:
            root = ET.parse(wf).getroot()
        except Exception as e:  # noqa: BLE001
            log(f"WARN parse {wf}: {e}")
            continue
        for w in root.iter("w"):          # only <w> elements are words
            if w.get("punc") == "true":
                continue
            txt = (w.text or "").strip()
            if not txt:
                continue
            try:
                s = float(w.get("starttime"))
                e = float(w.get("endtime"))
            except (TypeError, ValueError):
                continue
            yield s, e, txt, spk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True, help="AMI-diarization-setup root (for lists/)")
    ap.add_argument("--words", required=True, help="ami_public_manual/words dir")
    ap.add_argument("--audio", required=True, help="amicorpus root with <M>/audio/<M>.Mix-Headset.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--words-dir-out", required=True, help="dir for per-meeting *.words.jsonl sidecars")
    ap.add_argument("--splits", nargs="+", default=["test", "dev"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    annot, words, audio = Path(args.annot), Path(args.words), Path(args.audio)
    meetings = read_meetings(annot, args.splits)
    if args.limit:
        meetings = meetings[:args.limit]
    utts, missing = [], 0
    for m in meetings:
        wav = audio / m / "audio" / f"{m}.Mix-Headset.wav"
        ws = sorted(read_words(words, m), key=lambda x: x[0])
        if not ws:
            log(f"WARN no words for {m}")
            continue
        if not wav.exists():
            missing += 1
            log(f"WARN no audio for {m}: {wav}")
        ref = " ".join(w for _s, _e, w, _spk in ws)
        dur = max((e for _s, e, _w, _spk in ws), default=0.0)
        utts.append(Utt(utt_id=m, recording_id=m, dataset="ami", speaker_id="multi",
                        audio_path=str(wav), ref_text=ref, duration=round(dur, 2)))
        write_jsonl(Path(args.words_dir_out) / f"{m}.words.jsonl",
                    ({"start": s, "end": e, "word": w, "speaker": spk} for s, e, w, spk in ws))

    n = write_manifest(args.out, utts)
    total_words = sum(len(u.ref_text.split()) for u in utts)
    log(f"wrote {n} meetings ({total_words} words total) -> {args.out} ({missing} missing audio)")


if __name__ == "__main__":
    main()
