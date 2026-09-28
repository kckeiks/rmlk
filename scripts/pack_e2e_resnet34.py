#!/usr/bin/env python3
"""Pack ResNet34 e2e artifacts (opset-14 ONNX + dog.jpeg + sidecar).

Large weights stay out of git. The JPEG fixture is committed under
`runtime/tests/fixtures/e2e/resnet34/dog.jpeg`.

Example:

    PYTHONPATH=.venv-oracle/lib/python3.12/site-packages \\
      python3 scripts/pack_e2e_resnet34.py \\
        --model ~/Downloads/resnet34.onnx \\
        --out .cache/rmlk/e2e

Then either point the harness at the cache (default layout) after uploading
the same tree to Hugging Face, or set overrides:

    export RMLK_E2E_ASSET_RESNET34_MODEL=.../model.onnx
    export RMLK_E2E_ASSET_RESNET34_IMAGE=.../dog.jpeg
    export RMLK_E2E_ASSET_RESNET34_SIDECAR=.../sidecar.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "runtime/tests/fixtures/e2e/resnet34/dog.jpeg"
CASE_ID = "resnet34"
ARTIFACT_ID = "2026-09-27"
ORT_VERSION = "1.24.2"
TARGET_OPSET = 14


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    data = path.read_bytes()
    h.update(data)
    return h.hexdigest(), len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to ResNet34 ONNX (any opset; converted to 14)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Dog JPEG (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / ".cache/rmlk/e2e",
        help="Cache-style root (writes {case}/{artifact}/...)",
    )
    args = parser.parse_args()

    try:
        import onnx
        from onnx import version_converter
    except ImportError:
        print(
            "error: onnx not installed; use scripts/requirements-oracle.in",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.model.is_file():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    model = onnx.load(args.model)
    model = version_converter.convert_version(model, TARGET_OPSET)
    ir_version = int(model.ir_version)
    opsets = [(o.domain, o.version) for o in model.opset_import]
    ai_onnx = next((v for d, v in opsets if d in ("", "ai.onnx")), TARGET_OPSET)

    out_dir = args.out / CASE_ID / ARTIFACT_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.onnx"
    image_path = out_dir / "dog.jpeg"
    onnx.save(model, model_path)
    shutil.copy2(args.image, image_path)

    model_sha, model_bytes = sha256_file(model_path)
    image_sha, image_bytes = sha256_file(image_path)

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
            "image": {
                "path": "dog.jpeg",
                "sha256": image_sha,
                "bytes": image_bytes,
                "dtype": "jpeg",
            },
        },
        "notes": (
            "ImageNet-1k ResNet34; dog.jpeg preprocess is torchvision-style "
            "resize(shorter=256)+center-crop(224)+normalize. Expected top-1 "
            "for this fixture is class 207 (golden_retriever)."
        ),
    }
    sidecar_path = out_dir / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"wrote {out_dir}")
    print(f"  model.onnx  sha256={model_sha} bytes={model_bytes}")
    print(f"  dog.jpeg    sha256={image_sha} bytes={image_bytes}")
    print(f"  sidecar.json ir={ir_version} opset={ai_onnx}")
    print()
    print("Local overrides (skip Hub download):")
    print(f"  export RMLK_E2E_ASSET_RESNET34_MODEL={model_path.resolve()}")
    print(f"  export RMLK_E2E_ASSET_RESNET34_IMAGE={image_path.resolve()}")
    print(f"  export RMLK_E2E_ASSET_RESNET34_SIDECAR={sidecar_path.resolve()}")
    print()
    print("Or upload this directory tree to the Hub repo at")
    print(f"  {CASE_ID}/{ARTIFACT_ID}/")


if __name__ == "__main__":
    main()
