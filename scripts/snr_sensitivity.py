#!/usr/bin/env python3
"""SNR sensitivity: re-mix speech+music at multiple SNR levels, run CLAP detection."""
from __future__ import annotations
import argparse, csv, io, os, random, sys, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

def load16k(path_or_bytes):
    obj = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    y, sr = sf.read(obj, dtype="float32")
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(y, dtype="float32")

def mix(speech, music, snr_db):
    if len(music) < len(speech):
        music = np.tile(music, int(np.ceil(len(speech) / len(music))))
    music = music[:len(speech)]
    ps = np.mean(speech ** 2) + 1e-9
    pm = np.mean(music ** 2) + 1e-9
    g = np.sqrt(ps / (pm * 10 ** (snr_db / 10.0)))
    out = speech + g * music
    m = np.max(np.abs(out)) + 1e-9
    return (out / m * 0.98).astype("float32") if m > 1.0 else out.astype("float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-manifest", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--snr-levels", default="0,4,8,12,16")
    ap.add_argument("--out", default="results/stats/snr_sensitivity.csv")
    args = ap.parse_args()

    snr_levels = [int(x) for x in args.snr_levels.split(",")]
    random.seed(42); np.random.seed(42)

    from datasets import load_dataset, Audio
    ds = None
    for name in ("marsyas/gtzan", "sanchit-gandhi/gtzan", "confit/gtzan"):
        try:
            ds = load_dataset(name, split="train", trust_remote_code=True)
            print(f"Loaded {name} ({len(ds)})")
            break
        except Exception as e:
            print(f"FAIL {name}: {str(e)[:90]}")
    if ds is None:
        raise SystemExit("No GTZAN source")
    ak = next(k for k, v in ds.features.items() if isinstance(v, Audio))
    ds = ds.cast_column(ak, Audio(decode=False))
    idx = list(range(len(ds))); random.shuffle(idx)
    music = []
    for i in idx[:args.n]:
        ex = ds[i]
        a = ex[ak]
        y = load16k(a["bytes"]) if a.get("bytes") else load16k(a["path"])
        music.append(y)
    print(f"Loaded {len(music)} music clips")

    speech_rows = list(csv.DictReader(open(args.speech_manifest)))
    random.shuffle(speech_rows)
    speech_rows = speech_rows[:args.n]
    speech_audio = [load16k(r["audio_path"]) for r in speech_rows]
    print(f"Loaded {len(speech_audio)} speech clips")

    # Load CLAP with monkey-patch to avoid torchcodec
    import torch
    import torchaudio.transforms as _T
    from msclap import CLAP
    from msclap.CLAPWrapper import CLAPWrapper

    def _read_audio_sf(self, audio_path, resample=True):
        y, sr = sf.read(audio_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        t = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
        rr = self.args.sampling_rate
        if resample and rr != sr:
            t = _T.Resample(sr, rr)(t)
        return t, rr

    CLAPWrapper.read_audio = _read_audio_sf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    clap = CLAP(version="2023", use_cuda=(device == "cuda"))
    prompts = ["speech with music in the background", "speech without music"]
    emb_t = clap.get_text_embeddings(prompts)

    results = []
    for snr_db in snr_levels:
        correct_music, correct_clean = 0, 0
        for i in range(len(speech_audio)):
            sp = speech_audio[i]
            mus = music[i % len(music)]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, sp, 16000)
                emb_a = clap.get_audio_embeddings([tmp.name])
                sim = torch.nn.functional.cosine_similarity(emb_a, emb_t)
                if sim.argmax().item() == 1: correct_clean += 1
                os.unlink(tmp.name)

            mx = mix(sp, mus, snr_db)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, mx, 16000)
                emb_a = clap.get_audio_embeddings([tmp.name])
                sim = torch.nn.functional.cosine_similarity(emb_a, emb_t)
                if sim.argmax().item() == 0: correct_music += 1
                os.unlink(tmp.name)

            if (i + 1) % 20 == 0:
                print(f"  SNR={snr_db}dB: {i+1}/{len(speech_audio)}")

        n = len(speech_audio)
        acc_m = correct_music / n
        acc_c = correct_clean / n
        acc = (correct_music + correct_clean) / (2 * n)
        results.append({"snr_db": snr_db, "acc_music": f"{acc_m:.3f}",
                        "acc_clean": f"{acc_c:.3f}", "acc_overall": f"{acc:.3f}", "n": n})
        print(f"SNR={snr_db}dB: music={acc_m:.3f}, clean={acc_c:.3f}, overall={acc:.3f}")

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "acc_music", "acc_clean", "acc_overall", "n"])
        w.writeheader()
        w.writerows(results)
    print(f"Written: {outpath}")

if __name__ == "__main__":
    main()
