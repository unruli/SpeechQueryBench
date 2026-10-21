"""Score returned annotation sheets: Fleiss' kappa (IAA), majority consensus, and LLM-judge agreement.

Reads the 3 completed annotator_*.csv (each has code + discussed?), joins on `code` to the private key,
and reports: inter-annotator agreement (Fleiss' kappa), the majority-vote consensus human gold, and how
well the LLM judge matches it (Cohen's kappa + precision/recall/F1). Emits consensus_human_gold.jsonl
(recording_id -> concept ids) so sem_eval can be re-run on human gold.

Usage (.venvASR312, CPU):
    python scripts/score_human_gold.py --key annotation_kit/_key_PRIVATE_do_not_send.csv \
        --filled annotation_kit/annotator_A.csv annotation_kit/annotator_B.csv annotation_kit/annotator_C.csv \
        --out-gold results/ami/semantic/gold_human.jsonl
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict

from common import write_jsonl, log


def yn(v):
    v = (v or "").strip().lower()
    return 1 if v in ("yes", "y", "1", "true") else (0 if v in ("no", "n", "0", "false") else None)


def fleiss_kappa(table):  # table: list of [n_no, n_yes] per item, all with same n raters
    n = sum(table[0]); N = len(table)
    p = [sum(col) / (N * n) for col in zip(*table)]           # category marginals
    Pe = sum(pj * pj for pj in p)
    Pi = [(sum(x * x for x in row) - n) / (n * (n - 1)) for row in table]
    Pbar = sum(Pi) / N
    return (Pbar - Pe) / (1 - Pe) if (1 - Pe) else 1.0


def cohen_kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--filled", nargs="+", required=True)
    ap.add_argument("--out-gold", required=True)
    args = ap.parse_args()

    key = {r["code"]: r for r in csv.DictReader(open(args.key))}
    ann = []  # list of {code: 0/1}
    for path in args.filled:
        d = {}
        for r in csv.DictReader(open(path)):
            lab = yn(r.get("discussed?"))
            if lab is not None:
                d[r["code"]] = lab
        ann.append(d)
        miss = [c for c in key if c not in d]
        log(f"{path}: {len(d)}/{len(key)} labeled" + (f" (MISSING {len(miss)})" if miss else ""))

    codes = [c for c in key if all(c in d for d in ann)]
    table = [[sum(1 for d in ann if d[c] == 0), sum(1 for d in ann if d[c] == 1)] for c in codes]
    fk = fleiss_kappa(table)
    consensus = {c: (1 if sum(d[c] for d in ann) >= (len(ann) + 1) // 2 else 0) for c in codes}
    llm = {c: int(key[c]["llm_gold"]) for c in codes}

    hv = [consensus[c] for c in codes]; lv = [llm[c] for c in codes]
    tp = sum(l and h for l, h in zip(lv, hv)); fp = sum(l and not h for l, h in zip(lv, hv))
    fn = sum((not l) and h for l, h in zip(lv, hv))
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0

    print(f"\n=== HUMAN VALIDATION ({len(codes)} pairs, {len(ann)} annotators) ===")
    print(f"Inter-annotator agreement  Fleiss' kappa = {fk:.3f}")
    print(f"Consensus positive rate    = {sum(consensus.values())/len(codes):.3f}  "
          f"(LLM positive rate = {sum(lv)/len(codes):.3f})")
    print(f"LLM judge vs human gold    Cohen's kappa = {cohen_kappa(lv, hv):.3f}  "
          f"P={pr:.3f} R={rc:.3f} F1={f1:.3f}")

    # emit consensus human gold in gold.jsonl format (recording_id -> concept ids)
    by_meeting = defaultdict(list)
    for c in codes:
        if consensus[c]:
            by_meeting[key[c]["meeting_real"]].append(int(key[c]["concept_id"]))
    all_meetings = sorted({key[c]["meeting_real"] for c in codes})
    write_jsonl(args.out_gold, [{"recording_id": m, "concepts": sorted(by_meeting.get(m, []))}
                               for m in all_meetings])
    print(f"\nwrote consensus human gold -> {args.out_gold}  "
          f"(re-run sem_eval with --gold {args.out_gold} to re-score architectures)\n")


if __name__ == "__main__":
    main()
