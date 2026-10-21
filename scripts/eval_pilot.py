#!/usr/bin/env python3
"""Score the pilot: WER (+ S/D/I decomposition), query metrics, and the rank-reversal kill test.

Emits, next to the summary:
  * pilot_summary.csv   one row per system (adds sub/del/ins rates)
  * per_query.csv       one row per (system, query): lex_f1 / count_relerr / prop_mae + rarity
  * per_recording_wer.csv  one row per (system, recording): errors, ref_len   (for WER bootstrap)
The last two feed scripts/bootstrap.py (confidence intervals + paired system diffs).

Usage:
    python scripts/eval_pilot.py --manifest M.csv --workload W.jsonl \
        --asr-dir results/asr --queries-dir results/queries --out results/pilot_summary.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from common import read_manifest, read_jsonl, wer_sdi


def system_wer_stats(utts, hyps_path):
    """Return (stats dict with wer/sub/del/ins rates, per-recording {rec: [err, ref_len]})."""
    hyp = {r["utt_id"]: r.get("hyp_text", "") for r in read_jsonl(hyps_path)}
    s = d = i = ref = 0
    per_rec = defaultdict(lambda: [0, 0])
    for u in utts:
        us, ud, ui, un = wer_sdi(u.ref_text, hyp.get(u.utt_id, ""))
        s += us; d += ud; i += ui; ref += un
        per_rec[u.recording_id][0] += us + ud + ui
        per_rec[u.recording_id][1] += un
    r = ref or 1
    return {"wer": (s + d + i) / r, "sub": s / r, "del": d / r, "ins": i / r}, per_rec


def prf(pred: set, gold: set):
    inter = len(pred & gold)
    p = inter / len(pred) if pred else (1.0 if not gold else 0.0)
    r = inter / len(gold) if gold else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def load_answers(path):
    return {row["qid"]: row for row in read_jsonl(path)}


def score_system(name, gold, sysans, workload):
    """Return (aggregate metrics dict, list of per-query rows)."""
    lex_f1, lex_f1_by_rarity = [], defaultdict(list)
    cnt_relerr, prop_mae = [], []
    per_query = []
    q_by_id = {q["qid"]: q for q in workload}
    for qid, g in gold.items():
        s = sysans.get(qid, {})
        q = q_by_id[qid]
        row = {"system": name, "qid": qid, "family": q["family"], "op": q.get("op") or "",
               "rarity": q.get("rarity", ""), "lex_f1": "", "count_relerr": "", "prop_mae": ""}
        if q["family"] == "lexical":
            _, _, f = prf(set(s.get("result_set", [])), set(g.get("result_set", [])))
            lex_f1.append(f); lex_f1_by_rarity[q["rarity"]].append(f)
            row["lex_f1"] = f
        elif q.get("op") == "count":
            gc = g.get("count", 0)
            e = abs(s.get("count", 0) - gc) / max(gc, 1)
            cnt_relerr.append(e); row["count_relerr"] = e
        elif q.get("op") == "prop_by_speaker":
            gp, sp = g.get("prop", {}), s.get("prop", {})
            keys = set(gp) | set(sp)
            if keys:
                mae = sum(abs(gp.get(k, 0) - sp.get(k, 0)) for k in keys) / len(keys)
                prop_mae.append(mae); row["prop_mae"] = mae
        per_query.append(row)

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    agg = {
        "lexical_f1": mean(lex_f1),
        "f1_rare": mean(lex_f1_by_rarity.get("rare", [])),
        "f1_medium": mean(lex_f1_by_rarity.get("medium", [])),
        "f1_common": mean(lex_f1_by_rarity.get("common", [])),
        "count_relerr": mean(cnt_relerr),
        "prop_mae": mean(prop_mae),
    }
    return agg, per_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--queries-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    utts = read_manifest(args.manifest)
    workload = list(read_jsonl(args.workload))
    gold = load_answers(Path(args.queries_dir) / "reference.jsonl")

    systems, per_query_all, per_rec_all = [], [], []
    for ans_path in sorted(Path(args.queries_dir).glob("*.jsonl")):
        name = ans_path.stem
        if name == "reference":
            continue
        wer_stats, per_rec = system_wer_stats(utts, Path(args.asr_dir) / name / "hyps.jsonl")
        agg, per_query = score_system(name, gold, load_answers(ans_path), workload)
        systems.append({"system": name, **wer_stats, **agg})
        per_query_all.extend(per_query)
        for rec, (err, ref) in per_rec.items():
            per_rec_all.append({"system": name, "recording_id": rec, "errors": err, "ref_len": ref})

    systems.sort(key=lambda r: r["wer"])
    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)

    fields = ["system", "wer", "sub", "del", "ins", "lexical_f1", "f1_rare", "f1_medium",
              "f1_common", "count_relerr", "prop_mae"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in systems:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in fields})

    with open(outdir / "per_query.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "qid", "family", "op", "rarity",
                                          "lex_f1", "count_relerr", "prop_mae"])
        w.writeheader()
        w.writerows(per_query_all)

    with open(outdir / "per_recording_wer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "recording_id", "errors", "ref_len"])
        w.writeheader()
        w.writerows(per_rec_all)

    # --- report ---
    print("\n=== PILOT SUMMARY (sorted by WER, best first) ===")
    print(f"{'system':<12} {'WER':>7} {'sub':>6} {'del':>6} {'ins':>6} {'lexF1':>7} {'F1rare':>7} {'cntRelE':>8}")
    for r in systems:
        print(f"{r['system']:<12} {r['wer']:>7.3f} {r['sub']:>6.3f} {r['del']:>6.3f} {r['ins']:>6.3f} "
              f"{r['lexical_f1']:>7.3f} {r['f1_rare']:>7.3f} {r['count_relerr']:>8.3f}")

    print("\n=== KILL TEST: rank reversals (lower WER but worse query answers) ===")
    reversals = 0
    for a, b in combinations(systems, 2):
        if a["wer"] >= b["wer"]:
            continue
        for label, is_rev in (("F1", a["lexical_f1"] < b["lexical_f1"]),
                              ("F1(rare)", a["f1_rare"] < b["f1_rare"]),
                              ("count relErr", a["count_relerr"] > b["count_relerr"])):
            if is_rev:
                reversals += 1
                print(f"  REVERSAL: {a['system']} (WER {a['wer']:.3f}) worse than "
                      f"{b['system']} (WER {b['wer']:.3f}) on {label}")
    if reversals == 0:
        print("  none found — query quality tracks WER on this slice.")
    print(f"\n  total reversals: {reversals}  |  see bootstrap.py for whether the gaps are significant\n")


if __name__ == "__main__":
    main()
