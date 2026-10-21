#!/usr/bin/env python3
"""C1: cost-quality Pareto for the transcriber layer.

All seven systems transcribed the SAME AfriSpeech corpus (49 recordings, ~7.0 h audio), so their wall
times are directly comparable. We report the standard, load-insensitive speech cost metric -- real-time
factor RTF = processing_time / audio_time -- and an approximate $/audio-hour that normalizes for GPU
class (rate x RTF), then pair each with rare-keyword query utility (F1rare) and mark the Pareto frontier
(a system is dominated if another is at least as good on F1 AND at least as cheap). GPU rates are
illustrative on-demand figures (L4 $0.40/h, A40/RTX6000-class $0.80/h); the frontier is robust to them.

Usage (.venvASR312, CPU):
  python scripts/cost_quality.py --manifest data/manifests/afrispeech.csv \
    --per-query results/afrispeech/per_query.csv --timings results/cost/afrispeech_timings.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from common import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--per-query", required=True)
    ap.add_argument("--timings", required=True)
    args = ap.parse_args()

    total_s = sum(float(r.get("duration") or 0) for r in csv.DictReader(open(args.manifest)))

    f1 = defaultdict(list)
    for r in csv.DictReader(open(args.per_query)):
        if r["family"] == "lexical" and r["rarity"] == "rare" and r["lex_f1"]:
            f1[r["system"]].append(float(r["lex_f1"]))
    f1 = {s: sum(v) / len(v) for s, v in f1.items()}

    rows = []
    for t in csv.DictReader(open(args.timings)):
        s = t["system"]
        el = float(t["elapsed_s"])
        rtf = el / total_s
        usd_hr = rtf * float(t["rate_usd_hr"])  # $ per hour of audio processed
        rows.append({"system": s, "gpu": t["gpu"], "rtf": rtf, "xrealtime": 1 / rtf,
                     "usd_per_audio_hr": usd_hr, "f1rare": f1.get(s, float("nan"))})

    # Pareto frontier: minimize usd, maximize f1 (dominated if other has usd<= and f1>=, better in one)
    for r in rows:
        r["frontier"] = not any(
            o is not r and o["usd_per_audio_hr"] <= r["usd_per_audio_hr"] and o["f1rare"] >= r["f1rare"]
            and (o["usd_per_audio_hr"] < r["usd_per_audio_hr"] or o["f1rare"] > r["f1rare"])
            for o in rows)

    log(f"AfriSpeech audio = {total_s/3600:.2f} h over {sum(1 for _ in csv.DictReader(open(args.manifest)))} recs")
    print(f"\n=== C1 cost-quality (transcriber layer, AfriSpeech {total_s/3600:.1f} h) ===")
    print(f"{'system':<18}{'GPU':<14}{'RTF':>7}{'xRT':>7}{'$/audio-h':>11}{'F1rare':>8}  frontier")
    for r in sorted(rows, key=lambda x: x["usd_per_audio_hr"]):
        print(f"{r['system']:<18}{r['gpu']:<14}{r['rtf']:>7.4f}{r['xrealtime']:>7.0f}"
              f"{r['usd_per_audio_hr']:>11.4f}{r['f1rare']:>8.3f}  {'YES' if r['frontier'] else 'dominated'}")
    dom = [r["system"] for r in rows if not r["frontier"]]
    print(f"\nPareto-dominated (more expensive AND no better at querying): {', '.join(dom)}\n")


if __name__ == "__main__":
    main()
