"""Shared helpers for the EchoDB-Bench pilot: text normalization, WER, manifest/JSONL IO.

Kept dependency-light (stdlib only) so it runs anywhere. faster-whisper / pandas are only
imported by the scripts that need them.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator

# --- text normalization ---------------------------------------------------------------
# Prefer the STANDARD Whisper English text normalizer (numbers, contractions, spelling, casing)
# so WER is comparable to published Whisper numbers and keyword matching is canonical. Both gold
# and hypotheses pass through the same normalizer. Fall back to a crude uppercase/strip if the
# package is missing. Install: pip install whisper-normalizer
_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")

try:
    from whisper_normalizer.english import EnglishTextNormalizer as _ETN
    _whisper_norm = _ETN()
except Exception:
    try:
        from whisper.normalizers import EnglishTextNormalizer as _ETN  # openai-whisper
        _whisper_norm = _ETN()
    except Exception:
        _whisper_norm = None


def _crude(text: str) -> str:
    text = (text or "").upper()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def normalize(text: str) -> str:
    if _whisper_norm is not None:
        return _whisper_norm(text or "")
    return _crude(text)


def tokens(text: str) -> list[str]:
    return normalize(text).split()


# --- WER (word-level Levenshtein) -----------------------------------------------------
try:
    from rapidfuzz.distance import Levenshtein as _Lev  # C++-backed
except Exception:
    _Lev = None


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    if _Lev is not None:  # essential for long meeting-level transcripts (~5000 words)
        return _Lev.distance(ref, hyp)
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(ref_text: str, hyp_text: str) -> tuple[int, int]:
    """Return (errors, ref_len) so callers can micro-average correctly."""
    r, h = tokens(ref_text), tokens(hyp_text)
    return edit_distance(r, h), len(r)


def wer_sdi(ref_text: str, hyp_text: str) -> tuple[int, int, int, int]:
    """Return (substitutions, deletions, insertions, ref_len) from the alignment editops.

    deletions = reference words dropped by the ASR (recall loss);
    insertions = spurious words added by the ASR (over-transcription).
    This decomposition is what tests whether a higher-WER system is merely inserting more while
    actually recovering MORE reference content (fewer deletions) -> better query answers.
    """
    r, h = tokens(ref_text), tokens(hyp_text)
    if _Lev is not None:
        s = d = i = 0
        for op in _Lev.editops(r, h):
            if op.tag == "replace":
                s += 1
            elif op.tag == "delete":
                d += 1
            elif op.tag == "insert":
                i += 1
        return s, d, i, len(r)
    return edit_distance(r, h), 0, 0, len(r)  # fallback: no op decomposition


# --- data model -----------------------------------------------------------------------
@dataclass
class Utt:
    utt_id: str
    recording_id: str
    dataset: str
    speaker_id: str
    audio_path: str
    ref_text: str
    duration: float


MANIFEST_FIELDS = ["utt_id", "recording_id", "dataset", "speaker_id", "audio_path", "ref_text", "duration"]


def write_manifest(path: str | Path, utts: Iterable[Utt]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for u in utts:
            w.writerow(asdict(u))
            n += 1
    return n


def read_manifest(path: str | Path) -> list[Utt]:
    with open(path, newline="") as f:
        return [
            Utt(
                utt_id=r["utt_id"],
                recording_id=r["recording_id"],
                dataset=r["dataset"],
                speaker_id=r["speaker_id"],
                audio_path=r["audio_path"],
                ref_text=r["ref_text"],
                duration=float(r["duration"] or 0.0),
            )
            for r in csv.DictReader(f)
        ]


# --- JSONL ----------------------------------------------------------------------------
def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def log(*a):
    print(*a, file=sys.stderr, flush=True)
