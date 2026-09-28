# Compatibility profile

This document is the source of truth for which ONNX surface rmlk claims to
support and which ONNX Runtime build is the numerical oracle for tests. The
README may summarize the same numbers; if they disagree, this file wins until
the next bump commit updates both.

## Supported ONNX surface (rmlk)

| Axis | Value | Notes |
|------|-------|-------|
| ONNX IR version | 10 | Assumed from the README until a dedicated review. |
| Default opset domain | `ai.onnx` | |
| Supported opset version | 14 | Assumed from the README until a dedicated review. |

Models used for full-model e2e should be exported or converted so their
`opset_import` for `ai.onnx` is 14 (or lower, within what ORT and rmlk both
run). Raising the supported opset is a bump of this profile, not a silent
change in an example export.

## Numerical oracle (tests and scripts)

| Package | Pinned version | Role |
|---------|----------------|------|
| `onnx` | 1.21.0 | Build or inspect ONNX graphs in scripts. |
| `onnxruntime` | 1.24.2 | CPU oracle for fixture generation and host-side checks. |
| `onnxruntime-gpu` | 1.24.2 | Same ORT version for CUDA e2e on CUDA 12 machines. |

Pins live in [`scripts/requirements-oracle.in`](../scripts/requirements-oracle.in).
Install for script work with:

```bash
python3 -m pip install -r scripts/requirements-oracle.in
```

Or with `uv`:

```bash
uv venv .venv-oracle
uv pip install --python .venv-oracle/bin/python -r scripts/requirements-oracle.in
```

These pins deliberately track **ort 2.0.0-rc.12** / ORT **1.24.2**, the last
`ort` release that still ships **CUDA 12** prebuilts (rc.13 is CUDA 13-only).
Opset 14 models (Llama, converted ResNet) do not need ORT 1.28. Bump later
when the test machines move to CUDA 13.
### Regenerating graph integration goldens

The attention and conv-block fixtures under `runtime/tests/fixtures/` are
produced by building the same graphs as ONNX (opset 14, IR 10) and running them
with the pinned CPU ONNX Runtime. The Rust tests still build those graphs with
`GraphBuilder`; only the expected float blobs come from ORT.

```bash
python3 scripts/gen_graph_fixtures.py
# or: .venv-oracle/bin/python scripts/gen_graph_fixtures.py
```

The script exits if the installed `onnx` / `onnxruntime` versions do not match
this profile. After changing the graphs in `runtime/tests/graphs/`, regenerate
and commit any intentional fixture diffs.

### Official ONNX node cases

These tests compare rmlk to the official ONNX single-op suite that ships inside
the pinned `onnx` package. Case files are not stored in git. Discovery writes a
local cache manifest; the harness then classifies each case:

1. **Skipped (unknown op)** — the model uses an operator that is not in rmlk's
   `Op` set at all. The summary ranks those ops by how many cases they block.
2. **Dtype/cast gap** — every operator in the case is known, but the run fails
   because that op does not yet handle this element type or cast pair. Logged
   as `dtype_gap`; does not fail the test by itself.
3. **Failed** — wrong outputs or an unexpected error. Fails the test.
4. **Passed** — outputs match the suite within tolerance.

Requires a CUDA GPU. Ignored by default:

```bash
python3 scripts/discover_onnx_node_cases.py
cargo test -p rmlk-runtime --test onnx_node -- --ignored --nocapture
```

If the manifest is missing, the test runs the discover script automatically.
Override the cache directory with `RMLK_ONNX_NODE_CACHE` if needed.

The summary lists unknown-op skips and `dtype_gap` cases so you can see what
is not supported yet. Only hard `FAIL`s fail the test. A future won't-do
bucket can silence known permanent gaps if needed.

Known gaps: `ConstantOfShape` cases are skipped until the protobuf reader can
load packed tensor attributes (item 45). Softmax on axis 1 with rank-3 input
still fails until that path is implemented (needed for Nemotron ASR).

Use the CPU package for regenerating small graph goldens. Use
`onnxruntime-gpu` at the same version for manual full-model e2e on GPU.

Rust `ort` bindings used by in-process e2e must load this same ORT version:

| Crate | Pinned version | Features | Notes |
|-------|----------------|----------|-------|
| `ort` | `=2.0.0-rc.12` | `cuda` | Ships CUDA 12 and 13. Override with `ORT_CUDA_VERSION=12` if needed. |

Dev-dependency of `rmlk-runtime` (full-model e2e harness only).

Manual full-model run (CUDA 12 example):

```bash
export ORT_CUDA_VERSION=12
CUDA_COMPUTE_CAP=89 cargo test -p rmlk-runtime --test e2e_ort -- --ignored --nocapture
```

`ORT_CUDA_VERSION=12` selects ort's CUDA 12 prebuilts. `CUDA_COMPUTE_CAP`
is only needed when the cuda crate's `nvidia-smi` compute-cap probe fails
(set it to your GPU's sm version, e.g. `89` for Ada).## Bump checklist

Changing IR, opset, or the pinned ORT/onnx versions is one intentional change
set:

1. Update the tables in this file and the pins in `scripts/requirements-oracle.in`.
2. Update the README summary if it still mentions IR or opset.
3. Re-run official ONNX node cases for every op you claim (section 11 / item 53).
4. Regenerate ORT-based graph goldens (`scripts/gen_graph_fixtures.py`) and
   commit any fixture diffs that are still within expected tolerance policy.
5. Re-run the manual full-model e2e suite (ResNet, Llama) against the new oracle.
6. Refresh remote artifact sidecars so they record the new IR, opset, and ORT
   version, and update checksums if model bytes changed. Layout and env vars:
   [`e2e-artifacts.md`](e2e-artifacts.md).

Do not bump the oracle in the same commit as an unrelated kernel change. If a
kernel fix and an oracle bump both change fixtures, land them separately so
bisects stay readable.
