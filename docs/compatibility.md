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
| `onnxruntime` | 1.28.0 | CPU oracle for fixture generation and host-side checks. |
| `onnxruntime-gpu` | 1.28.0 | Same ORT version for CUDA e2e when a CUDA 13-capable machine is available. |

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

### Node-suite results baseline

Passed case names (and optionally `dtype_gap` names) are recorded under
`runtime/tests/baselines/onnx_node-<onnx_version>.json`. The default GPU run
diffs against that file and fails if any previously `ok` case regressed to
`dtype_gap` or `FAIL`. New failures that were never in the baseline still fail
the suite on their own.

After an intentional improvement or onnx pin bump, rewrite the baseline:

```bash
RMLK_ONNX_NODE_BLESS=1 cargo test -p rmlk-runtime --test onnx_node -- --ignored --nocapture
```

Commit the updated JSON in the same change as the improvement or pin bump.
Host-only unit tests for the baseline format and regression check live in the
same `onnx_node` test binary and do not need a GPU.

Known gaps: `ConstantOfShape` cases are skipped until the protobuf reader can
load packed tensor attributes (item 45). Softmax on axis 1 with rank-3 input
still fails until that path is implemented (needed for Nemotron ASR).

Use the CPU package for regenerating small graph goldens. Use
`onnxruntime-gpu` at the same version for manual full-model e2e on GPU. If the
local CUDA toolkit is older than what that GPU wheel requires, keep the oracle
**version** pin and install a matching older GPU wheel only as a temporary
local override; do not change the profile pin without following the bump
checklist below.

Rust `ort` bindings used by in-process e2e must load this same ORT version:

| Crate | Pinned version | Features |
|-------|----------------|----------|
| `ort` | `=2.0.0-rc.13` | `cuda` |

Dev-dependency of `rmlk-runtime` (full-model e2e harness only).

## Bump checklist

Changing IR, opset, or the pinned ORT/onnx versions is one intentional change
set:

1. Update the tables in this file and the pins in `scripts/requirements-oracle.in`.
2. Update the README summary if it still mentions IR or opset.
3. Re-run official ONNX node cases for every op you claim (section 11 / item 53),
   once that harness exists. Re-bless
   `runtime/tests/baselines/onnx_node-<version>.json` if the pin change is
   intentional (`RMLK_ONNX_NODE_BLESS=1`).
4. Regenerate ORT-based graph goldens (`scripts/gen_graph_fixtures.py`) and
   commit any fixture diffs that are still within expected tolerance policy.
5. Re-run the manual full-model e2e suite (ResNet, Llama) against the new oracle.
6. Refresh remote artifact sidecars so they record the new IR, opset, and ORT
   version, and update checksums if model bytes changed. Layout and env vars:
   [`e2e-artifacts.md`](e2e-artifacts.md).

Do not bump the oracle in the same commit as an unrelated kernel change. If a
kernel fix and an oracle bump both change fixtures, land them separately so
bisects stay readable.
