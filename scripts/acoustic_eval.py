#!/usr/bin/env python3
"""Score acoustic-context architectures: background-music detection (binary) + genre.

Balanced set (N clean + N music), so binary chance = .5 and genre chance = .1 (10 genres). Reports
binary accuracy + music-class F1, and genre accuracy over the music clips. Audio-native (CLAP) should
clear both; transcript-based should sit at chance (words are identical across clean/music twins).

Usage:
    python scripts/acoustic_eval.py --pred clap=results/acoustic/clap.jsonl --pred textllm=results/acoustic/textllm.jsonl
"""
from __future__ import annotations

import argparse

from common import read_jsonl


def score(rows):
    n = len(rows)
    acc = sum(r["has_music_pred"] == r["has_music_gold"] for r in rows) / n
    tp = sum(r["has_music_pred"] == 1 and r["has_music_gold"] == 1 for r in rows)
    fp = sum(r["has_music_pred"] == 1 and r["has_music_gold"] == 0 for r in rows)
    fn = sum(r["has_music_pred"] == 0 and r["has_music_gold"] == 1 for r in rows)
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
    music = [r for r in rows if r["has_music_gold"] == 1]
    gen_acc = sum(r["genre_pred"] == r["genre_gold"] for r in music) / len(music) if music else 0.0
    return acc, f1, gen_acc, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[], help="name=path")
    args = ap.parse_args()
    print("\n=== ACOUSTIC-CONTEXT: background-music detection + genre ===")
    print(f"{'architecture':<22}{'binAcc':>8}{'musicF1':>9}{'genreAcc':>10}")
    print(f"{'chance':<22}{0.5:>8.3f}{'--':>9}{0.1:>10.3f}")
    for spec in args.pred:
        name, path = spec.split("=", 1)
        acc, f1, gen, n = score(list(read_jsonl(path)))
        print(f"{name:<22}{acc:>8.3f}{f1:>9.3f}{gen:>10.3f}")
    print("\n(clean/music twins share the same words => transcript blind; audio-native reads the ambient sound.)\n")


if __name__ == "__main__":
    main()
