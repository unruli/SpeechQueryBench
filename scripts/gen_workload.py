#!/usr/bin/env python3
"""Generate the pilot query workload from reference transcripts.

Two families (the ones most likely to decouple from WER):
  * lexical  : SELECT recordings WHERE transcript CONTAINS <keyword>   (stratified by rarity)
  * aggregate: COUNT recordings mentioning <keyword>  +  proportion GROUP BY speaker

Rarity is defined by document frequency (df = # recordings whose REFERENCE contains the word),
relative to the number of recordings R. Gold answers are derived from references at eval time
(scripts/run_queries.py), so the workload only stores query specs — keeping gold and prediction
computed by the identical code path.

Usage:
    python scripts/gen_workload.py --manifest M.csv --out workload/queries.jsonl \
        --per-stratum 40 --seed 0
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

from common import read_manifest, tokens, write_jsonl, log

STOP = set(
    "THE A AN AND OR BUT IF THEN ELSE OF TO IN ON AT BY FOR WITH AS IS ARE WAS WERE BE BEEN "
    "BEING HE SHE IT THEY WE YOU I HIM HER THEM US HIS HERS ITS THEIR OUR MY YOUR THIS THAT "
    "THESE THOSE WHICH WHO WHOM WHOSE WHAT WHEN WHERE WHY HOW NOT NO SO THAN THAT'S HAD HAVE "
    "HAS DO DID DOES WOULD COULD SHOULD WILL SHALL MAY MIGHT MUST CAN AM ALL ANY SOME SUCH "
    "THERE HERE UP DOWN OUT ONE TWO INTO FROM ABOUT OVER THEM'S UPON".split()
)


def recording_texts(utts) -> dict[str, str]:
    buf = defaultdict(list)
    for u in utts:
        buf[u.recording_id].append(u.ref_text)
    return {rid: " ".join(t) for rid, t in buf.items()}


def doc_freq(rec_texts: dict[str, str]) -> dict[str, int]:
    df = defaultdict(int)
    for txt in rec_texts.values():
        for w in set(tokens(txt)):
            df[w] += 1
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-stratum", type=int, default=40)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    utts = read_manifest(args.manifest)
    rec_texts = recording_texts(utts)
    R = len(rec_texts)
    df = doc_freq(rec_texts)

    cand = [
        w for w, d in df.items()
        if d >= 1 and len(w) >= args.min_len and w.isalpha() and w not in STOP
    ]
    rare, medium, common = [], [], []
    for w in cand:
        d = df[w]
        if d <= max(1, int(0.02 * R)):
            rare.append(w)
        elif d <= max(2, int(0.15 * R)):
            medium.append(w)
        else:
            common.append(w)
    log(f"R={R} recordings | vocab candidates={len(cand)} | rare={len(rare)} med={len(medium)} common={len(common)}")

    queries = []
    qid = 0
    for stratum, pool in (("rare", rare), ("medium", medium), ("common", common)):
        rng.shuffle(pool)
        for w in pool[: args.per_stratum]:
            # lexical selection
            queries.append({"qid": qid, "family": "lexical", "keyword": w,
                            "df": df[w], "selectivity": round(df[w] / R, 4), "rarity": stratum})
            qid += 1
            # aggregate count over the same keyword
            queries.append({"qid": qid, "family": "aggregate", "op": "count", "keyword": w,
                            "df": df[w], "selectivity": round(df[w] / R, 4), "rarity": stratum})
            qid += 1
            # aggregate proportion grouped by speaker
            queries.append({"qid": qid, "family": "aggregate", "op": "prop_by_speaker", "keyword": w,
                            "df": df[w], "selectivity": round(df[w] / R, 4), "rarity": stratum})
            qid += 1

    n = write_jsonl(args.out, queries)
    log(f"wrote {n} queries ({sum(q['family']=='lexical' for q in queries)} lexical, "
        f"{sum(q['family']=='aggregate' for q in queries)} aggregate) -> {args.out}")


if __name__ == "__main__":
    main()
