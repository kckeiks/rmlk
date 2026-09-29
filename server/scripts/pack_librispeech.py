#!/usr/bin/env python3
"""Pack LibriSpeech WAVs for rmlk-server ASR tests.

Downloads the matching OpenSLR tarball (cached under `.cache/rmlk/asr/openslr/`),
converts FLACs to 16 kHz mono PCM16 WAV under `.cache/rmlk/asr/librispeech/`,
and optionally rewrites a committed job manifest (references + checksums).

Flags mirror Hugging Face `openslr/librispeech_asr`:
  --config  Hub config name (`clean` / `other`)
  --split   Hub split (`test` / …)
  --id      Optional comma-separated utterance ids (Hub `id` field).
            Omit to pack the entire config/split.
  --manifest
            Path to a job manifest to rewrite (only when packing a subset,
            or when explicitly requested).

Dependencies: ffmpeg, Python 3 stdlib.

Usage (from repo root):
  # shared pool: all of clean/test
  python3 server/scripts/pack_librispeech.py --config clean --split test

  # small e2e suite (once)
  python3 server/scripts/pack_librispeech.py --config clean --split test \\
    --id 6930-75918-0000,6930-75918-0001,6930-75918-0002 \\
    --manifest server/tests/fixtures/asr/manifests/e2e.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import wave
from pathlib import Path

DATASET = "openslr/librispeech_asr"
DEFAULT_CONFIG = "clean"
DEFAULT_SPLIT = "test"
DEFAULT_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"

# OpenSLR resource 12 archives keyed by (config, split).
OPENSLR_ARCHIVES: dict[tuple[str, str], tuple[str, str]] = {
    # (url filename, path prefix inside the tarball)
    ("clean", "test"): (
        "test-clean.tar.gz",
        "LibriSpeech/test-clean",
    ),
    ("other", "test"): (
        "test-other.tar.gz",
        "LibriSpeech/test-other",
    ),
}
OPENSLR_BASE_URL = "https://www.openslr.org/resources/12"

SCHEMA_VERSION = 2
CHUNK_SAMPLES = 8960
SAMPLE_RATE = 16_000
DEFAULT_MANIFEST = (
    "server/tests/fixtures/asr/manifests/e2e.json"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def parse_id_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    if not ids:
        sys.exit("--id was empty")
    return ids


def archive_for(config: str, split: str) -> tuple[str, str]:
    key = (config, split)
    if key not in OPENSLR_ARCHIVES:
        supported = ", ".join(f"{c}/{s}" for c, s in sorted(OPENSLR_ARCHIVES))
        raise SystemExit(
            f"unsupported --config/--split {config}/{split}; "
            f"supported: {supported}"
        )
    return OPENSLR_ARCHIVES[key]


def ensure_openslr_tar(cache_root: Path, filename: str) -> Path:
    dest = cache_root / "openslr" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"using cached tarball {dest}")
        return dest
    url = f"{OPENSLR_BASE_URL}/{filename}"
    print(f"downloading {url} → {dest} …")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    req = urllib.request.Request(url, headers={"User-Agent": "rmlk-pack-librispeech/1"})
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)
    return dest


def parse_utt_id(utt_id: str) -> tuple[str, str, str]:
    parts = utt_id.split("-")
    if len(parts) != 3:
        raise SystemExit(f"expected speaker-chapter-utt id, got {utt_id!r}")
    return parts[0], parts[1], parts[2]


def flac_member(prefix: str, utt_id: str) -> str:
    speaker, chapter, _ = parse_utt_id(utt_id)
    return f"{prefix}/{speaker}/{chapter}/{utt_id}.flac"


def trans_member(prefix: str, utt_id: str) -> str:
    speaker, chapter, _ = parse_utt_id(utt_id)
    return f"{prefix}/{speaker}/{chapter}/{speaker}-{chapter}.trans.txt"


def list_all_utt_ids(tar: tarfile.TarFile, prefix: str) -> list[str]:
    ids: list[str] = []
    pat = re.compile(
        rf"^{re.escape(prefix)}/(\d+)/(\d+)/(\d+-\d+-\d+)\.flac$"
    )
    for name in tar.getnames():
        m = pat.match(name)
        if m:
            ids.append(m.group(3))
    ids.sort()
    if not ids:
        raise SystemExit(f"no FLAC utterances under {prefix}/ in tarball")
    return ids


def load_references(
    tar: tarfile.TarFile, prefix: str, ids: list[str]
) -> dict[str, str]:
    needed_trans = {trans_member(prefix, i) for i in ids}
    by_utt: dict[str, str] = {}
    for member_name in needed_trans:
        try:
            f = tar.extractfile(member_name)
        except KeyError as err:
            raise SystemExit(f"missing {member_name} in OpenSLR tarball") from err
        if f is None:
            raise SystemExit(f"could not read {member_name}")
        for line in f.read().decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt, _, text = line.partition(" ")
            by_utt[utt] = text
    missing = [i for i in ids if i not in by_utt]
    if missing:
        raise SystemExit(f"utterance ids missing from trans.txt: {missing}")
    return {i: by_utt[i] for i in ids}


def flac_to_wav(flac_path: Path, wav_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(flac_path),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(wav_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def job_name_from_manifest(path: Path) -> str:
    # manifests/e2e.json → e2e
    return path.stem


def write_manifest(
    manifest_path: Path,
    *,
    job: str,
    config: str,
    split: str,
    revision: str,
    clips: list[dict],
) -> None:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "job": job,
        "dataset": DATASET,
        "dataset_config": config,
        "dataset_split": split,
        "dataset_revision": revision,
        "sample_rate_hz": SAMPLE_RATE,
        "channels": 1,
        "pcm": "s16le",
        "chunk_samples": CHUNK_SAMPLES,
        "clips": clips,
        "notes": (
            f"{job} job over {DATASET} ({config}/{split}, OpenSLR). "
            "References are official LibriSpeech transcripts from *.trans.txt. "
            "Regenerate with server/scripts/pack_librispeech.py."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


def pack(
    *,
    config: str,
    split: str,
    revision: str,
    ids: list[str] | None,
    manifest_path: Path | None,
) -> None:
    root = repo_root()
    cache_root = root / ".cache" / "rmlk" / "asr"
    wav_dir = cache_root / "librispeech"
    wav_dir.mkdir(parents=True, exist_ok=True)

    filename, prefix = archive_for(config, split)
    tar_path = ensure_openslr_tar(cache_root, filename)

    clips: list[dict] = []
    with tarfile.open(tar_path, "r:gz") as tar, tempfile.TemporaryDirectory(
        prefix="rmlk-librispeech-"
    ) as tmp:
        tmp_dir = Path(tmp)
        selected = ids if ids is not None else list_all_utt_ids(tar, prefix)
        print(f"packing {len(selected)} utterance(s) from {config}/{split} …")
        refs = load_references(tar, prefix, selected)

        for i, uid in enumerate(selected, start=1):
            member = flac_member(prefix, uid)
            try:
                src = tar.extractfile(member)
            except KeyError as err:
                raise SystemExit(f"missing {member} in OpenSLR tarball") from err
            if src is None:
                raise SystemExit(f"could not read {member}")
            flac_path = tmp_dir / f"{uid}.flac"
            with flac_path.open("wb") as out:
                shutil.copyfileobj(src, out)

            wav_name = f"{uid}.wav"
            wav_path = wav_dir / wav_name
            flac_to_wav(flac_path, wav_path)
            clip = {
                "id": uid,
                "wav": wav_name,
                "reference": refs[uid],
                "sha256": sha256_file(wav_path),
                "bytes": wav_path.stat().st_size,
                "duration_s": round(duration_s(wav_path), 2),
            }
            clips.append(clip)
            if ids is not None or i % 50 == 0 or i == len(selected):
                print(f"  [{i}/{len(selected)}] {uid} ({clip['duration_s']}s)")

    if manifest_path is not None:
        job = job_name_from_manifest(manifest_path)
        write_manifest(
            manifest_path,
            job=job,
            config=config,
            split=split,
            revision=revision,
            clips=clips,
        )
    elif ids is not None:
        print(
            "packed subset WAVs but no --manifest given; "
            "pass --manifest to rewrite a committed job manifest"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Hub dataset config (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Hub dataset split (default: {DEFAULT_SPLIT})",
    )
    p.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hub dataset revision recorded in the manifest",
    )
    p.add_argument(
        "--id",
        dest="id_list",
        default=None,
        help="Comma-separated Hub utterance ids; omit to pack the full config/split",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help=(
            "Job manifest to rewrite (e.g. server/tests/fixtures/asr/manifests/e2e.json). "
            f"Default when --id is set: {DEFAULT_MANIFEST}"
        ),
    )
    args = p.parse_args()
    ids = parse_id_list(args.id_list)

    manifest_path: Path | None = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = repo_root() / manifest_path
    elif ids is not None:
        manifest_path = repo_root() / DEFAULT_MANIFEST

    pack(
        config=args.config,
        split=args.split,
        revision=args.revision,
        ids=ids,
        manifest_path=manifest_path,
    )


if __name__ == "__main__":
    main()
