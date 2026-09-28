#!/usr/bin/env python3
"""Pack Llama-3.2-3B-Instruct e2e artifacts (ONNX + tokenizer + prompt + sidecar).

Large weights stay out of git. By default this script **symlinks** `model.onnx`
and `model.onnx_data` into the cache tree (no 12 GiB copy). The prompt fixture
is committed under `runtime/tests/fixtures/e2e/llama3.2/prompt.txt`.

Example:

    PYTHONPATH=.venv-oracle/lib/python3.12/site-packages \\
      python3 scripts/pack_e2e_llama32.py \\
        --model-dir ~/Models/Llama-3.2-3B-Instruct \\
        --out .cache/rmlk/e2e

Then either point the harness at the cache (default layout) after uploading
the same tree to Hugging Face, or set overrides that point at the original
model directory (external data must sit beside `model.onnx`):

    export RMLK_E2E_ASSET_LLAMA3_2_MODEL=.../onnx/model.onnx
    export RMLK_E2E_ASSET_LLAMA3_2_MODEL_DATA=.../onnx/model.onnx_data
    export RMLK_E2E_ASSET_LLAMA3_2_TOKENIZER=.../tokenizer.json
    export RMLK_E2E_ASSET_LLAMA3_2_PROMPT=.../prompt.txt
    export RMLK_E2E_ASSET_LLAMA3_2_SIDECAR=.../sidecar.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT / "runtime/tests/fixtures/e2e/llama3.2/prompt.txt"
CASE_ID = "llama3.2"
ARTIFACT_ID = "2026-09-27"
ORT_VERSION = "1.24.2"
TARGET_OPSET = 14


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Llama-3.2-3B-Instruct root (contains onnx/model.onnx + tokenizer.json)",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT,
        help=f"UTF-8 prompt (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / ".cache/rmlk/e2e",
        help="Cache-style root (writes {case}/{artifact}/...)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy model.onnx / model.onnx_data instead of symlinking",
    )
    parser.add_argument(
        "--skip-data-hash",
        action="store_true",
        help="Skip hashing model.onnx_data (writes bytes=0 sha placeholder; Hub publish needs a real hash)",
    )
    args = parser.parse_args()

    try:
        import onnx
    except ImportError:
        print(
            "error: onnx not installed; use scripts/requirements-oracle.in",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir = args.model_dir.expanduser().resolve()
    model_src = model_dir / "onnx" / "model.onnx"
    data_src = model_dir / "onnx" / "model.onnx_data"
    tok_src = model_dir / "tokenizer.json"

    for path, label in [
        (model_src, "model.onnx"),
        (data_src, "model.onnx_data"),
        (tok_src, "tokenizer.json"),
        (args.prompt, "prompt"),
    ]:
        if not path.is_file():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    model = onnx.load(str(model_src), load_external_data=False)
    ir_version = int(model.ir_version)
    opsets = [(o.domain, o.version) for o in model.opset_import]
    ai_onnx = next((v for d, v in opsets if d in ("", "ai.onnx")), TARGET_OPSET)
    if int(ai_onnx) > TARGET_OPSET:
        print(
            f"error: model opset {ai_onnx} > supported {TARGET_OPSET}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = args.out / CASE_ID / ARTIFACT_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.onnx"
    data_path = out_dir / "model.onnx_data"
    tok_path = out_dir / "tokenizer.json"
    prompt_path = out_dir / "prompt.txt"

    link_or_copy(model_src, model_path, copy=args.copy)
    link_or_copy(data_src, data_path, copy=args.copy)
    shutil.copy2(tok_src, tok_path)
    shutil.copy2(args.prompt, prompt_path)

    print("hashing model.onnx …", flush=True)
    model_sha, model_bytes = sha256_file(model_path)
    if args.skip_data_hash:
        data_sha = "0" * 64
        data_bytes = data_src.stat().st_size
        print(
            "warning: --skip-data-hash set; sidecar model_data sha256 is a placeholder",
            file=sys.stderr,
        )
    else:
        print("hashing model.onnx_data (large) …", flush=True)
        data_sha, data_bytes = sha256_file(data_path)
    tok_sha, tok_bytes = sha256_file(tok_path)
    prompt_sha, prompt_bytes = sha256_file(prompt_path)

    sidecar = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "artifact_id": ARTIFACT_ID,
        "onnx_ir_version": ir_version,
        "onnx_opset": int(ai_onnx),
        "ort_version": ORT_VERSION,
        "files": {
            "model": {
                "path": "model.onnx",
                "sha256": model_sha,
                "bytes": model_bytes,
            },
            "model_data": {
                "path": "model.onnx_data",
                "sha256": data_sha,
                "bytes": data_bytes,
            },
            "tokenizer": {
                "path": "tokenizer.json",
                "sha256": tok_sha,
                "bytes": tok_bytes,
            },
            "prompt": {
                "path": "prompt.txt",
                "sha256": prompt_sha,
                "bytes": prompt_bytes,
                "dtype": "utf8",
            },
        },
        "notes": (
            "Llama-3.2-3B-Instruct fp32 ONNX with external model.onnx_data; "
            "chat-template greedy decode; short prompt fixture; max_new_tokens=8 "
            "in the harness case."
        ),
    }
    sidecar_path = out_dir / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"wrote {out_dir}")
    print(f"  model.onnx       sha256={model_sha} bytes={model_bytes}")
    print(f"  model.onnx_data  sha256={data_sha} bytes={data_bytes}")
    print(f"  tokenizer.json   sha256={tok_sha} bytes={tok_bytes}")
    print(f"  prompt.txt       sha256={prompt_sha} bytes={prompt_bytes}")
    print(f"  sidecar.json ir={ir_version} opset={ai_onnx}")
    print()
    print("Local overrides (skip Hub download):")
    # Prefer pointing model at the original dir so external data resolves without
    # relying on cache symlinks surviving across machines.
    print(f"  export RMLK_E2E_ASSET_LLAMA3_2_MODEL={model_src}")
    print(f"  export RMLK_E2E_ASSET_LLAMA3_2_MODEL_DATA={data_src}")
    print(f"  export RMLK_E2E_ASSET_LLAMA3_2_TOKENIZER={tok_path.resolve()}")
    print(f"  export RMLK_E2E_ASSET_LLAMA3_2_PROMPT={prompt_path.resolve()}")
    print(f"  export RMLK_E2E_ASSET_LLAMA3_2_SIDECAR={sidecar_path.resolve()}")
    print()
    print("Or upload this directory tree to the Hub repo at")
    print(f"  {CASE_ID}/{ARTIFACT_ID}/")


if __name__ == "__main__":
    main()
