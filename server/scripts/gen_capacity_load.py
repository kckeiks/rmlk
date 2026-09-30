#!/usr/bin/env python3
"""Generate Dirigo-compatible capacity_load WAVs into the rmlk testdata cache.

Mirrors `dirigo/benchmarks/datasets/download.py` (`generate_capacity_load`) so
latency benches can use the same PCM as Dirigo's STT capacity harness.

Writes under `.cache/rmlk/testdata/` (or `RMLK_TESTDATA_CACHE`) and prints
paths + SHA-256 for the capacity job manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / ".cache" / "rmlk" / "testdata"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "tests" / "manifests" / "capacity.json"
SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 8960


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    frames = (pcm * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


def _tone_sweep(duration_s: float, sample_rate: int, f0: float, f1: float) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    freqs = np.linspace(f0, f1, len(t))
    phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
    return (0.2 * np.sin(phase)).astype(np.float32)


def _speechish(duration_s: float, sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration_s)
    noise = rng.normal(0, 0.15, n).astype(np.float32)
    t = np.arange(n) / sample_rate
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    gated = noise * env
    for start in range(0, n, sample_rate):
        gated[start : start + int(0.08 * sample_rate)] *= 0.05
    return gated.astype(np.float32)


def generate(out_dir: Path) -> list[tuple[str, Path, float]]:
    """Return list of (id, path, duration_s)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("loop_tone_a", _tone_sweep(4.0, SAMPLE_RATE, 300, 900), 4.0),
        ("loop_tone_b", _tone_sweep(4.0, SAMPLE_RATE, 500, 1200), 4.0),
        ("loop_speechish", _speechish(6.0, SAMPLE_RATE, seed=42), 6.0),
    ]
    written: list[tuple[str, Path, float]] = []
    for clip_id, audio, duration_s in specs:
        path = out_dir / f"{clip_id}.wav"
        _write_wav(path, audio, SAMPLE_RATE)
        written.append((clip_id, path, duration_s))
    return written


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(clips: list[tuple[str, Path, float]], path: Path) -> None:
    entries = []
    for clip_id, wav_path, duration_s in clips:
        entries.append(
            {
                "id": clip_id,
                "wav": wav_path.name,
                "reference": "",
                "sha256": sha256_file(wav_path),
                "bytes": wav_path.stat().st_size,
                "duration_s": duration_s,
            }
        )
    manifest = {
        "schema_version": 2,
        "job": "capacity",
        "dataset": "dirigo/capacity_load",
        "dataset_config": "synthetic",
        "dataset_split": "capacity",
        "dataset_revision": "dirigo-generate_capacity_load",
        "sample_rate_hz": SAMPLE_RATE,
        "channels": 1,
        "pcm": "s16le",
        "chunk_samples": CHUNK_SAMPLES,
        "clips": entries,
        "notes": (
            "Same synthetic loops as Dirigo benchmarks/datasets/capacity_load "
            "(generate_capacity_load in download.py). Not an accuracy corpus. "
            "Regenerate with server/scripts/gen_capacity_load.py."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("RMLK_TESTDATA_CACHE", DEFAULT_CACHE)),
        help="WAV output directory (default: RMLK_TESTDATA_CACHE or .cache/rmlk/testdata)",
    )
    p.add_argument(
        "--write-manifest",
        action="store_true",
        help=f"Rewrite {MANIFEST_PATH.relative_to(ROOT)}",
    )
    args = p.parse_args()

    clips = generate(args.out_dir)
    print(f"Wrote {len(clips)} capacity_load WAVs to {args.out_dir}")
    for clip_id, path, _ in clips:
        digest = sha256_file(path)
        print(f"  {clip_id}: {path.name} sha256={digest} bytes={path.stat().st_size}")

    if args.write_manifest:
        write_manifest(clips, MANIFEST_PATH)
        print(f"Wrote manifest {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
