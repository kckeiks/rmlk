#!/usr/bin/env python3
"""Generate binary f32 fixtures for runtime/tests/graphs integration tests.

Expected outputs come from the pinned ONNX Runtime in
`scripts/requirements-oracle.in` (see `docs/compatibility.md`). Inputs and
constants are still chosen here so they match the Rust GraphBuilder tests.

Install the oracle, then regenerate:

    python3 -m pip install -r scripts/requirements-oracle.in
    python3 scripts/gen_graph_fixtures.py

Re-run after changing the reference graphs in `runtime/tests/graphs/`.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime" / "tests" / "fixtures"

# Match docs/compatibility.md / scripts/requirements-oracle.in
EXPECTED_ONNX = "1.21.0"
EXPECTED_ORT = "1.28.0"
OPSET = 14


def write_f32(path: Path, values: np.ndarray | list[float]) -> None:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(f"<{arr.size}f", *arr.tolist()))
    print(f"wrote {path.relative_to(ROOT)} ({arr.size} f32)")


def check_oracle_versions() -> None:
    onnx_ver = onnx.__version__
    ort_ver = ort.__version__
    if onnx_ver != EXPECTED_ONNX or ort_ver != EXPECTED_ORT:
        print(
            "error: oracle package versions do not match docs/compatibility.md\n"
            f"  onnx:        got {onnx_ver}, expected {EXPECTED_ONNX}\n"
            f"  onnxruntime: got {ort_ver}, expected {EXPECTED_ORT}\n"
            "Install with: python3 -m pip install -r scripts/requirements-oracle.in",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"oracle ok: onnx=={onnx_ver} onnxruntime=={ort_ver} opset={OPSET}")


def run_onnx(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    onnx.checker.check_model(model)
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    outs = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    return {name: np.asarray(val) for name, val in zip(names, outs)}


def gen_attention() -> None:
    # Mirrors runtime/tests/graphs/attention.rs
    q = np.array([float(i + 1) * 0.1 for i in range(8)], dtype=np.float32).reshape(1, 2, 4)
    k = np.array([float(i + 1) * 0.05 for i in range(8)], dtype=np.float32).reshape(1, 2, 4)
    v = np.array([float(i + 1) * 0.25 for i in range(8)], dtype=np.float32).reshape(1, 2, 4)
    scale = np.array(2.0, dtype=np.float32)

    nodes = [
        helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["q", "kt"], ["scores"]),
        helper.make_node("Div", ["scores", "scale"], ["scaled"]),
        helper.make_node("Trilu", ["scaled"], ["masked"], upper=0),
        helper.make_node("Softmax", ["masked"], ["probs"], axis=-1),
        helper.make_node("MatMul", ["probs", "v"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "attention_head",
        [
            helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 2, 4]),
            helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 2, 4]),
            helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 2, 4]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 4])],
        [numpy_helper.from_array(scale, name="scale")],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=10,
    )
    out = run_onnx(model, {"q": q, "k": k, "v": v})["out"]

    write_f32(OUT / "attention_q.f32", q)
    write_f32(OUT / "attention_k.f32", k)
    write_f32(OUT / "attention_v.f32", v)
    write_f32(OUT / "attention_out.f32", out)


def gen_conv_block() -> None:
    # Mirrors runtime/tests/graphs/conv_block.rs
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    w = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    ).reshape(2, 1, 3, 3)
    fc = np.array([1.0, 0.0, 0.5, 0.0, 1.0, -0.5], dtype=np.float32).reshape(2, 3)
    flat_shape = np.array([1, 2], dtype=np.int64)

    nodes = [
        helper.make_node(
            "Conv",
            ["x", "w"],
            ["conv_y"],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            dilations=[1, 1],
        ),
        helper.make_node("Relu", ["conv_y"], ["relu_y"]),
        helper.make_node(
            "MaxPool",
            ["relu_y"],
            ["pool_y"],
            kernel_shape=[2, 2],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
        ),
        helper.make_node("GlobalAveragePool", ["pool_y"], ["gap_y"]),
        helper.make_node("Reshape", ["gap_y", "flat_shape"], ["flat"]),
        helper.make_node("Gemm", ["flat", "fc"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "conv_block",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])],
        [
            numpy_helper.from_array(w, name="w"),
            numpy_helper.from_array(fc, name="fc"),
            numpy_helper.from_array(flat_shape, name="flat_shape"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=10,
    )
    out = run_onnx(model, {"x": x})["out"]

    write_f32(OUT / "conv_block_x.f32", x)
    write_f32(OUT / "conv_block_w.f32", w)
    write_f32(OUT / "conv_block_fc.f32", fc)
    write_f32(OUT / "conv_block_out.f32", out)


def main() -> None:
    check_oracle_versions()
    gen_attention()
    gen_conv_block()


if __name__ == "__main__":
    main()
