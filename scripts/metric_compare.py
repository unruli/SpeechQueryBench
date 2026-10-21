#!/usr/bin/env python3
"""Does a recall-oriented error rate predict query utility better than WER?

Per (system, recording), whole-transcript error rates from the S/D/I alignment:
  WER = (S+D+I)/N     insertions weighted like everything else (the standard metric)
  CER = (S+D)/N       content error rate: insertions discounted entirely
  DEL = D/N           pure deletion (recall-loss) rate

Per-recording query-utility outcome, over the workload's lexical keywords K that truly occur in the
recording's reference (TP=preserved, FN=dropped, FP=spuriously inserted):
  recall_err = FN/(TP+FN)        recall-oriented (a dropped keyword = missed match / undercount)
  f1_err     = 1 - F1            includes precision (a keyword INSERTION = false match) -> the
                                 harder, less-favorable test for a recall-oriented metric

Claim supported if |Spearman(CER, err)| > |Spearman(WER, err)|. We bootstrap the correlation
DIFFERENCE (resample recordings) and flag it SIGNIFICANT if the 95% CI excludes 0. We also print
system-level means so you can see whether CER re-ranks systems into query-utility order where WER
fails (the reversal fix).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from common import read_manifest, read_jsonl, tokens, wer_sdi


def ref_rec_text(utts):
    buf = defaultdict(list)
    for u in utts:
        buf[u.recording_id].append(u.ref_text)
    return {r: " ".join(t) for r, t in buf.items()}


def hyp_rec_text(hyps_path):
    buf = defaultdict(list)
    for r in read_jsonl(hyps_path):
        buf[r["recording_id"]].append(r.get("hyp_text", ""))
    return {rid: " ".join(t) for rid, t in buf.items()}


def lexical_keywords(workload):
    return {q["keyword"] for q in read_jsonl(workload) if q.get("family") == "lexical"}


def build_rows(utts, workload, asr_dir):
    ref = ref_rec_text(utts)
    K = lexical_keywords(workload)
    rows = []
    for hp in sorted(Path(asr_dir).glob("*/hyps.jsonl")):
        sysname = hp.parent.name
        hyp = hyp_rec_text(hp)
        for rid, rtext in ref.items():
            htext = hyp.get(rid, "")
            reftok, hyptok = set(tokens(rtext)), set(tokens(htext))
            Kref = K & reftok
            if not Kref:               # no query keyword in this reference -> outcome undefined
                continue
            Khyp = K & hyptok
            tp = len(Kref & Khyp); fn = len(Kref - Khyp); fp = len(Khyp - Kref)
            recall = tp / (tp + fn) if (tp + fn) else 1.0
            prec = tp / (tp + fp) if (tp + fp) else 1.0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
            s, d, i, n = wer_sdi(rtext, htext)
            n = n or 1
            rows.append({"system": sysname,
                         "WER": (s + d + i) / n, "CER": (s + d) / n, "DEL": d / n,
                         "recall_err": (fn / (tp + fn)) if (tp + fn) else 0.0,
                         "f1_err": 1.0 - f1})
    return rows


def boot_corr_diff(m_a, m_b, y, B=2000, seed=0):
    """Bootstrap |rho(a,y)| - |rho(b,y)| by resampling rows; return (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(B, n))
    diffs = []
    for b in range(B):
        j = idx[b]
        ra = spearmanr(m_a[j], y[j]).correlation
        rb = spearmanr(m_b[j], y[j]).correlation
        if ra == ra and rb == rb:  # skip nan (constant resample)
            diffs.append(abs(ra) - abs(rb))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--label", default="dataset")
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()

    rows = build_rows(read_manifest(args.manifest), args.workload, args.asr_dir)
    col = lambda k: np.array([r[k] for r in rows])
    W, C, D = col("WER"), col("CER"), col("DEL")

    print(f"\n######## {args.label}: recall-oriented metric vs WER at predicting query utility ########")
    print(f"(n={len(rows)} system×recording points, pooled over systems)")
    for ycol in ("recall_err", "f1_err"):
        y = col(ycol)
        print(f"\n-- outcome: keyword {ycol} --")
        for m in ("WER", "CER", "DEL"):
            print(f"   Spearman({m}, {ycol}) = {spearmanr(col(m), y).correlation:+.3f}")
        for nm, mcol in (("CER", C), ("DEL", D)):
            mean, lo, hi = boot_corr_diff(mcol, W, y, B=args.B)
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
            print(f"   |rho({nm})| - |rho(WER)| = {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {sig}")

    print("\n-- system-level means (does CER re-rank into query-utility order where WER fails?) --")
    bysys = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("WER", "CER", "DEL", "recall_err", "f1_err"):
            bysys[r["system"]][k].append(r[k])
    print(f"   {'system':<10} {'WER':>7} {'CER':>7} {'DEL':>7} {'recallErr':>10} {'f1Err':>7}")
    for s in sorted(bysys, key=lambda s: np.mean(bysys[s]["WER"])):
        b = bysys[s]
        print(f"   {s:<10} {np.mean(b['WER']):>7.3f} {np.mean(b['CER']):>7.3f} {np.mean(b['DEL']):>7.3f} "
              f"{np.mean(b['recall_err']):>10.3f} {np.mean(b['f1_err']):>7.3f}")


if __name__ == "__main__":
    main()
