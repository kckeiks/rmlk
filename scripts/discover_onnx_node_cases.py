#!/usr/bin/env python3
"""Discover official ONNX node test cases and write a local cache manifest.

Walks `onnx/backend/test/data/node` from the pinned `onnx` package, records each
case name, op types, and tensor dtypes, and writes a JSON manifest. Case bytes
stay in the onnx install; they are not committed to git.

Requires the oracle pins:

    python3 -m pip install -r scripts/requirements-oracle.in
    python3 scripts/discover_onnx_node_cases.py

Environment:

    RMLK_ONNX_NODE_CACHE  Override cache directory (default: .cache/rmlk/onnx-node/<ver>)
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ONNX = "1.21.0"


def dtype_name(code: int) -> str:
    from onnx import TensorProto

    try:
        return TensorProto.DataType.Name(code)
    except ValueError:
        return f"UNKNOWN_{code}"


def collect_dtypes(case_dir: Path, model) -> list[str]:
    from onnx import TensorProto

    codes: set[int] = set()
    for vi in list(model.graph.input) + list(model.graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type:
            codes.add(vi.type.tensor_type.elem_type)
    for init in model.graph.initializer:
        if init.data_type:
            codes.add(init.data_type)
    for pb_path in case_dir.glob("test_data_set_*/*.pb"):
        tensor = TensorProto()
        tensor.ParseFromString(pb_path.read_bytes())
        if tensor.data_type:
            codes.add(tensor.data_type)
    return sorted(dtype_name(c) for c in codes)


def main() -> None:
    try:
        import onnx
    except ImportError:
        print(
            "error: onnx is not installed. "
            "Install with: python3 -m pip install -r scripts/requirements-oracle.in",
            file=sys.stderr,
        )
        sys.exit(1)

    if onnx.__version__ != EXPECTED_ONNX:
        print(
            f"error: onnx=={onnx.__version__}, expected {EXPECTED_ONNX} "
            "(see docs/compatibility.md)",
            file=sys.stderr,
        )
        sys.exit(1)

    node_root = Path(onnx.__file__).resolve().parent / "backend" / "test" / "data" / "node"
    if not node_root.is_dir():
        print(f"error: ONNX node data not found at {node_root}", file=sys.stderr)
        sys.exit(1)

    cache = Path(
        os.environ.get(
            "RMLK_ONNX_NODE_CACHE",
            ROOT / ".cache" / "rmlk" / "onnx-node" / EXPECTED_ONNX,
        )
    )
    cache.mkdir(parents=True, exist_ok=True)

    cases = []
    op_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    errors = 0

    for case_dir in sorted(p for p in node_root.iterdir() if p.is_dir()):
        model_path = case_dir / "model.onnx"
        if not model_path.is_file():
            continue
        # Expanded graphs are multi-node rewrites; skip for node conformance.
        if "expanded" in case_dir.name:
            continue
        try:
            model = onnx.load(model_path)
            ops = sorted({n.op_type for n in model.graph.node if n.op_type})
            dtypes = collect_dtypes(case_dir, model)
        except Exception as exc:  # noqa: BLE001 — discovery must continue
            print(f"warn: skip {case_dir.name}: {exc}", file=sys.stderr)
            errors += 1
            continue
        for op in ops:
            op_counts[op] += 1
        for dtype in dtypes:
            dtype_counts[dtype] += 1
        cases.append({"name": case_dir.name, "ops": ops, "dtypes": dtypes})

    manifest = {
        "onnx_version": EXPECTED_ONNX,
        "manifest_version": 2,
        "node_data_root": str(node_root),
        "cases": cases,
        "op_counts": dict(sorted(op_counts.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
    }
    out = cache / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    (cache / "node_data_root").write_text(str(node_root) + "\n")

    print(f"wrote {out}")
    print(
        f"cases={len(cases)} unique_ops={len(op_counts)} "
        f"unique_dtypes={len(dtype_counts)} load_errors={errors}"
    )
    print(f"node_data_root={node_root}")


if __name__ == "__main__":
    main()
