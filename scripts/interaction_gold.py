#!/usr/bin/env python3
"""Build gold for the temporal/interaction query family from AMI RTTM speaker turns.

These queries operate on speaker STRUCTURE, which a plain transcript does not contain -- so the
dominant transcribe-then-search architecture cannot answer them at all. Gold per meeting:
  n_speakers      distinct speakers
  n_turns         number of speaker-turn segments
  n_changes       speaker changes along the timeline (consecutive segments with a different speaker)
  has_overlap     any two segments overlap in time (bool)
  overlap_frac    overlapped duration / total speech duration
  dominant_share  max per-speaker talk time / total talk time

RTTM line: SPEAKER <meeting> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>

Usage:
    python scripts/interaction_gold.py --rttm-root "$AMI_ANNOT/word_and_vocalsounds/rttms" \
        --splits test dev --out results/ami/interaction/gold.jsonl
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from common import write_jsonl, log


def parse_rttm(path):
    segs = []  # (start, end, speaker)
    for line in open(path):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            try:
                s = float(p[3]); d = float(p[4])
            except ValueError:
                continue
            segs.append((s, s + d, p[7]))
    return segs


def meeting_features(segs):
    if not segs:
        return None
    segs = sorted(segs)
    speakers = {sp for _s, _e, sp in segs}
    talk = {}
    for s, e, sp in segs:
        talk[sp] = talk.get(sp, 0.0) + (e - s)
    total_talk = sum(talk.values())
    # speaker changes along time-ordered segments
    order = [sp for _s, _e, sp in segs]
    n_changes = sum(1 for i in range(1, len(order)) if order[i] != order[i - 1])
    # overlap: sweep for any interval intersection + overlapped duration (union-based approx)
    overlap_dur = 0.0
    has_overlap = False
    ivs = sorted((s, e) for s, e, _sp in segs)
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s < cur_e:                      # overlaps current merged interval
            has_overlap = True
            overlap_dur += min(cur_e, e) - s
            cur_e = max(cur_e, e)
        else:
            cur_s, cur_e = s, e
    return {
        "n_speakers": len(speakers),
        "n_turns": len(segs),
        "n_changes": n_changes,
        "has_overlap": bool(has_overlap),
        "overlap_frac": round(overlap_dur / total_talk, 4) if total_talk else 0.0,
        "dominant_share": round(max(talk.values()) / total_talk, 4) if total_talk else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rttm-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["test", "dev"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for split in args.splits:
        for f in sorted(glob.glob(str(Path(args.rttm_root) / split / "*.rttm"))):
            mid = Path(f).stem
            feat = meeting_features(parse_rttm(f))
            if feat:
                rows.append({"meeting": mid, "split": split, **feat})
    write_jsonl(args.out, rows)
    import statistics as st
    log(f"wrote {len(rows)} meetings -> {args.out}")
    if rows:
        log(f"  median speakers={st.median(r['n_speakers'] for r in rows)}, "
            f"median turns={int(st.median(r['n_turns'] for r in rows))}, "
            f"overlap in {sum(r['has_overlap'] for r in rows)}/{len(rows)} meetings")


if __name__ == "__main__":
    main()
