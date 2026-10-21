#!/usr/bin/env python3
"""C1 extended to semantic architectures: cost as a function of QUERY VOLUME (index-once vs scan-per-query).

The semantic families expose a database tradeoff the transcriber layer hides: transcribe-then-search
architectures pay a ONE-TIME indexing cost (transcription, reusable across every query and family) plus a
cheap per-query operator, while audio-native QA re-runs the audio model on ALL audio for EVERY query.
We model $ to answer Q distinct concept queries over the 12-meeting AMI set (6.46 h, 783 chunks), using
measured GPU real-time factors and stage micro-benchmarks, and mark the Pareto frontier at each Q.

Rates: RTF = processing/audio time. GPU $/h illustrative (L4 $0.40, A40/RTX6000 $0.80).
"""
from __future__ import annotations

# ---- measured / derived constants ----
D = 23245.7            # s, total audio over 12 meetings
NCHUNKS = 783
RATE_L4, RATE_BIG = 0.40, 0.80

RTF_ASR = 0.136        # measured, large-v3 on L4 (C1)
RTF_QA = 0.117         # measured, Qwen2-Audio on A40/RTX6000 (C1); QA scans all chunks
RTF_CLAP = 0.0015      # measured: 34s / 23245s audio, CLAP on L4
EMBED_BUILD_S = 3.86   # measured, all-mpnet on L4: 4.93 ms/passage x 783 passages
EMBED_QUERY_S = 0.081  # measured, all-mpnet query encode on L4 (80.7 ms)
LLM_S_PER_MTG = 2.66   # measured, Qwen2.5-7B multi-label call, 1 meeting (8452 tok in) on L4
NMTG = 12

def gpu_cost(rtf, rate):            # $ to process the whole collection once at this rtf/rate
    return rtf * D * rate / 3600.0

C_ASR = gpu_cost(RTF_ASR, RATE_L4)         # one-time transcript index (shared by all text archs + families)
C_CLAP = gpu_cost(RTF_CLAP, RATE_L4)       # one-time audio (CLAP) index
C_QA_PERQ = gpu_cost(RTF_QA, RATE_BIG)     # per query: audio-LLM scans all audio (Qwen2-Audio, A40/RTX6000)
C_LLM_PERQ = NMTG * LLM_S_PER_MTG * RATE_L4 / 3600.0   # per query-batch: 12 Qwen2.5-7B calls (fits on L4)
C_EMBED_BUILD = EMBED_BUILD_S * RATE_L4 / 3600.0        # one-time embed index (L4); tiny
C_EMBED_PERQ = EMBED_QUERY_S * RATE_L4 / 3600.0         # per query: encode 1 query (near-zero)

# architecture: (fixed one-time $, per-query $, F1, indexed?)
ARCH = {
    "keyword":          (C_ASR,               0.0,        0.648, True),
    "text-embedding":   (C_ASR + C_EMBED_BUILD, C_EMBED_PERQ, 0.775, True),
    "ASR+text-LLM":     (C_ASR,               C_LLM_PERQ, 0.880, False),
    "speech-LLM-QA":    (0.0,                 C_QA_PERQ,  0.629, False),
    "CLAP":             (C_CLAP,              0.0,        0.629, True),
    "hybrid(retr->QA)": (C_ASR,               0.30 * C_QA_PERQ, 0.682, False),  # verify ~30% of chunks
}


def total(name, Q):
    fx, pq, _, _ = ARCH[name]
    return fx + pq * Q


def pareto(Q):
    costs = {n: total(n, Q) for n in ARCH}
    f1 = {n: ARCH[n][2] for n in ARCH}
    front = {}
    for n in ARCH:
        front[n] = not any(m != n and costs[m] <= costs[n] and f1[m] >= f1[n]
                           and (costs[m] < costs[n] or f1[m] > f1[n]) for m in ARCH)
    return costs, front


def main():
    print(f"Constants: C_ASR(index)=${C_ASR:.3f}  C_CLAP(index)=${C_CLAP:.3f}  "
          f"QA/query=${C_QA_PERQ:.3f}  textLLM/query=${C_LLM_PERQ:.4f}\n")
    for Q in (1, 10, 100):
        costs, front = pareto(Q)
        print(f"=== Q={Q} distinct queries : $ to answer over 12 meetings ===")
        print(f"{'architecture':<18}{'$total':>9}{'F1':>7}  frontier")
        for n in sorted(ARCH, key=lambda x: costs[x]):
            print(f"{n:<18}{costs[n]:>9.3f}{ARCH[n][2]:>7.3f}  {'YES' if front[n] else 'dominated'}")
        print()
    # crossover: text-embedding vs ASR+text-LLM (both accurate); QA always dominated
    fxE, _, _, _ = ARCH["text-embedding"]; fxL, pqL, _, _ = ARCH["ASR+text-LLM"]
    print(f"text-embedding cost is flat at ${fxE:.3f}; ASR+text-LLM = ${fxL:.3f}+${pqL:.4f}*Q "
          f"-> exceeds embedding immediately and grows; speech-LLM-QA = ${C_QA_PERQ:.3f}*Q (worst F1).")


if __name__ == "__main__":
    main()
