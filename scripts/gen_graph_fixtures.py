#!/usr/bin/env python3
"""Generate binary f32 fixtures for runtime/tests/graphs integration tests.

Pure Python (stdlib only). Re-run after changing the reference graphs:

    python3 scripts/gen_graph_fixtures.py
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime" / "tests" / "fixtures"


def write_f32(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))
    print(f"wrote {path.relative_to(ROOT)} ({len(values)} f32)")


def matmul(a: list[float], a_shape: tuple[int, ...], b: list[float], b_shape: tuple[int, ...]) -> list[float]:
    """Batched matmul for ranks 2–3 with numpy-like broadcasting of leading dims."""
    def as_batched(shape):
        if len(shape) == 2:
            return 1, shape[0], shape[1], False
        if len(shape) == 3:
            return shape[0], shape[1], shape[2], True
        raise ValueError(shape)

    ba, ma, ka, a3 = as_batched(a_shape)
    bb, kb, nb, b3 = as_batched(b_shape)
    assert ka == kb
    batch = max(ba, bb)
    out = []
    for bi in range(batch):
        ai = 0 if ba == 1 else bi
        bj = 0 if bb == 1 else bi
        for i in range(ma):
            for j in range(nb):
                s = 0.0
                for k in range(ka):
                    s += a[ai * ma * ka + i * ka + k] * b[bj * kb * nb + k * nb + j]
                out.append(s)
    return out


def transpose_021(x: list[float], n: int, a: int, b: int) -> list[float]:
    """Transpose axes (0,2,1) on [n,a,b] -> [n,b,a]."""
    out = [0.0] * (n * a * b)
    for i in range(n):
        for j in range(a):
            for k in range(b):
                out[i * b * a + k * a + j] = x[i * a * b + j * b + k]
    return out


def softmax_last(x: list[float], rows: int, cols: int) -> list[float]:
    out = []
    for r in range(rows):
        row = x[r * cols : (r + 1) * cols]
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        s = sum(exps)
        out.extend(e / s for e in exps)
    return out


def tril_batch(x: list[float], batch: int, n: int) -> list[float]:
    out = x[:]
    for b in range(batch):
        base = b * n * n
        for i in range(n):
            for j in range(n):
                if j > i:
                    out[base + i * n + j] = 0.0
    return out


def gen_attention() -> None:
    # Q,K,V: [1, 2, 4]
    q = [float(i + 1) * 0.1 for i in range(8)]
    k = [float(i + 1) * 0.05 for i in range(8)]
    v = [float(i + 1) * 0.25 for i in range(8)]
    scale = 2.0  # sqrt(4)

    kt = transpose_021(k, 1, 2, 4)  # [1,4,2]
    scores = matmul(q, (1, 2, 4), kt, (1, 4, 2))  # [1,2,2]
    scores = [s / scale for s in scores]
    scores = tril_batch(scores, 1, 2)
    scores = softmax_last(scores, 2, 2)
    out = matmul(scores, (1, 2, 2), v, (1, 2, 4))  # [1,2,4]

    write_f32(OUT / "attention_q.f32", q)
    write_f32(OUT / "attention_k.f32", k)
    write_f32(OUT / "attention_v.f32", v)
    write_f32(OUT / "attention_out.f32", out)


def conv2d(x, x_shape, w, w_shape):
    """Valid (no pad) NCHW conv, stride 1."""
    n, c_in, h, w_in = x_shape
    c_out, c_w, kh, kw = w_shape
    assert c_in == c_w
    oh, ow = h - kh + 1, w_in - kw + 1
    out = []
    for ni in range(n):
        for oc in range(c_out):
            for oy in range(oh):
                for ox in range(ow):
                    s = 0.0
                    for ic in range(c_in):
                        for ky in range(kh):
                            for kx in range(kw):
                                xv = x[
                                    ni * c_in * h * w_in
                                    + ic * h * w_in
                                    + (oy + ky) * w_in
                                    + (ox + kx)
                                ]
                                wv = w[oc * c_in * kh * kw + ic * kh * kw + ky * kw + kx]
                                s += xv * wv
                    out.append(s)
    return out, (n, c_out, oh, ow)


def relu(x: list[float]) -> list[float]:
    return [max(0.0, v) for v in x]


def max_pool_2x2_stride2(x, shape):
    n, c, h, w = shape
    assert h % 2 == 0 and w % 2 == 0
    oh, ow = h // 2, w // 2
    out = []
    for ni in range(n):
        for ci in range(c):
            for oy in range(oh):
                for ox in range(ow):
                    vals = []
                    for ky in range(2):
                        for kx in range(2):
                            vals.append(
                                x[
                                    ni * c * h * w
                                    + ci * h * w
                                    + (oy * 2 + ky) * w
                                    + (ox * 2 + kx)
                                ]
                            )
                    out.append(max(vals))
    return out, (n, c, oh, ow)


def global_avg_pool(x, shape):
    n, c, h, w = shape
    out = []
    spatial = h * w
    for ni in range(n):
        for ci in range(c):
            base = ni * c * spatial + ci * spatial
            out.append(sum(x[base : base + spatial]) / spatial)
    return out, (n, c, 1, 1)


def gemm(a, a_shape, b, b_shape):
    return matmul(a, a_shape, b, b_shape)


def gen_conv_block() -> None:
    # x: [1,1,4,4], w: [2,1,3,3] -> conv [1,2,2,2] -> relu -> maxpool [1,2,1,1]
    # -> gap [1,2,1,1] -> reshape [1,2] -> gemm with [2,3] -> [1,3]
    x = [float(i) for i in range(16)]
    w = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,  # filter 0: identity-ish diagonal
        0.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,  # filter 1: anti-diagonal
    ]
    fc = [
        1.0, 0.0, 0.5,
        0.0, 1.0, -0.5,
    ]  # [2, 3]

    y, sh = conv2d(x, (1, 1, 4, 4), w, (2, 1, 3, 3))
    y = relu(y)
    y, sh = max_pool_2x2_stride2(y, sh)
    y, sh = global_avg_pool(y, sh)
    flat = y  # [1,2,1,1] -> treat as [1,2]
    out = gemm(flat, (1, 2), fc, (2, 3))

    write_f32(OUT / "conv_block_x.f32", x)
    write_f32(OUT / "conv_block_w.f32", w)
    write_f32(OUT / "conv_block_fc.f32", fc)
    write_f32(OUT / "conv_block_out.f32", out)


def main() -> None:
    gen_attention()
    gen_conv_block()


if __name__ == "__main__":
    main()
