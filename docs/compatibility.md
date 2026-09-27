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

The harness discovers cases from this pin's installed `onnx` package (under
`onnx/backend/test/data/node`), writes a local cache manifest, and runs every
case whose op types are all supported by rmlk. Case bytes are not committed;
bumping `onnx` and re-running discovery picks up the new suite. The summary
lists ops that blocked the most cases so unsupported coverage is visible.

```bash
python3 scripts/discover_onnx_node_cases.py
cargo test -p rmlk-runtime --test onnx_node -- --ignored --nocapture
```

If the manifest is missing, the test tries to run the discover script itself.
Override the cache directory with `RMLK_ONNX_NODE_CACHE` if needed.

`ConstantOfShape` cases are skipped for now: packed `TENSOR` attributes abort
quick-protobuf on unaligned reads (item 45). Softmax cases that need axis 1 on
rank-3 inputs remain runnable and will fail until that support lands (Nemotron
ASR).

Use the CPU package for regenerating small graph goldens. Use
`onnxruntime-gpu` at the same version for manual full-model e2e on GPU. If the
local CUDA toolkit is older than what that GPU wheel requires, keep the oracle
**version** pin and install a matching older GPU wheel only as a temporary
local override; do not change the profile pin without following the bump
checklist below.

Rust `ort` bindings used by in-process e2e (when added) must load this same
ORT version. Record the crate pin next to these Python pins in the commit that
introduces the harness.

## Bump checklist

Changing IR, opset, or the pinned ORT/onnx versions is one intentional change
set:

1. Update the tables in this file and the pins in `scripts/requirements-oracle.in`.
2. Update the README summary if it still mentions IR or opset.
3. Re-run official ONNX node cases for every op you claim (section 11 / item 53),
   once that harness exists.
4. Regenerate ORT-based graph goldens (`scripts/gen_graph_fixtures.py`) and
   commit any fixture diffs that are still within expected tolerance policy.
5. Re-run the manual full-model e2e suite (ResNet, Llama) against the new oracle.
6. Refresh remote artifact sidecars so they record the new IR, opset, and ORT
   version, and update checksums if model bytes changed.

Do not bump the oracle in the same commit as an unrelated kernel change. If a
kernel fix and an oracle bump both change fixtures, land them separately so
bisects stay readable.
