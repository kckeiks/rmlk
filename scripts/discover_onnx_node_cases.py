#!/usr/bin/env python3
"""Discover official ONNX node test cases and write a local cache manifest.

Walks `onnx/backend/test/data/node` from the pinned `onnx` package, records each
case name and the op types in its `model.onnx`, and writes a JSON manifest.
Case bytes stay in the onnx install (or a copy under the cache); they are not
committed to git.

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
        except Exception as exc:  # noqa: BLE001 — discovery must continue
            print(f"warn: skip {case_dir.name}: {exc}", file=sys.stderr)
            errors += 1
            continue
        for op in ops:
            op_counts[op] += 1
        cases.append({"name": case_dir.name, "ops": ops})

    manifest = {
        "onnx_version": EXPECTED_ONNX,
        "node_data_root": str(node_root),
        "cases": cases,
        "op_counts": dict(sorted(op_counts.items())),
    }
    out = cache / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    # Convenience pointer for the harness default lookup.
    (cache / "node_data_root").write_text(str(node_root) + "\n")

    print(f"wrote {out}")
    print(f"cases={len(cases)} unique_ops={len(op_counts)} load_errors={errors}")
    print(f"node_data_root={node_root}")


if __name__ == "__main__":
    main()
