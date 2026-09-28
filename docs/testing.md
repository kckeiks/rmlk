# Testing guide

One place for how rmlk is tested, what each suite covers, and the commands to
run them. Use this when refactoring: host + GPU unit/graph tests for day-to-day
work; ONNX node suite and full-model e2e before larger merges.

Related docs:

| Doc | Role |
|-----|------|
| [`compatibility.md`](compatibility.md) | ONNX IR/opset pins and ORT oracle versions |
| [`e2e-artifacts.md`](e2e-artifacts.md) | Full-model artifact layout, sidecars, Hub / overrides |

## Philosophy

- **Numerical oracle** is pinned ONNX Runtime (not a second home-grown
  reference). Pins live in `scripts/requirements-oracle.in` and
  [`compatibility.md`](compatibility.md).
- **CUDA only** for runtime execution. Host crates (`rmlk-schema`,
  `rmlk-graph`, `rocky`) test without a GPU.
- Graphs go through production entry points (`GraphBuilder` →
  `Builder::from_graph` / `from_onnx_path`). Tests do not hand-build runtime
  types.
- Full-model e2e is **manual** (local or future on-demand CI). It is not a
  required PR check. Every e2e run compares to ORT; there is no bless file.

## Suite map

| Suite | Where | GPU? | Default `cargo test`? | CI today |
|-------|-------|------|------------------------|----------|
| Host crate unit tests | `schema`, `graph`, `rocky` | No | Yes | Yes (`host-tests`) |
| Asset resolver unit tests | `runtime/tests/e2e_assets` | No | Yes | Via workspace if GPU job on; otherwise run locally |
| CUDA op unit tests (`OpTest`) | `runtime/src/providers/cuda/backend/*.rs` | Yes | Yes (needs GPU) | Optional self-hosted `gpu-tests` |
| Multi-op graph + serialize | `runtime/tests/graphs` | Yes | Yes (needs GPU) | Optional GPU job |
| Official ONNX node suite | `runtime/tests/onnx_node` | Yes | **Ignored** | Manual |
| Full-model e2e (ResNet, Llama) | `runtime/tests/e2e_ort` | Yes + ORT CUDA | **Ignored** | Manual |

Also on every PR (no GPU required for fmt; check installs a CUDA toolkit for
compile):

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo check --workspace --all-targets
cargo test -p rmlk-schema -p rmlk-graph -p rocky
```

See `.github/workflows/ci.yml`.

## Machine prerequisites

### CUDA / driver

- A CUDA GPU and a toolkit matching what you build against (this repo’s CI
  uses CUDA 12.6 for compile).
- If the `cuda` crate cannot probe compute capability via `nvidia-smi`, set
  it explicitly (Ada / RTX 40-class example):

```bash
export CUDA_COMPUTE_CAP=89
```

### Python oracle (scripts)

```bash
# Option A: system / user pip
python3 -m pip install -r scripts/requirements-oracle.in

# Option B: project venv (recommended)
uv venv .venv-oracle
uv pip install --python .venv-oracle/bin/python -r scripts/requirements-oracle.in
# then: PYTHONPATH=.venv-oracle/lib/python3.12/site-packages python3 …
#   or: .venv-oracle/bin/python …
```

Pinned today: `onnx==1.21.0`, `onnxruntime==1.24.2`. For GPU ORT in Python
separately, use `onnxruntime-gpu==1.24.2` (do not mix CPU and GPU wheels in
one env). Details: [`compatibility.md`](compatibility.md).

### Rust ORT for in-process e2e

Full-model e2e uses the `ort` crate (`=2.0.0-rc.12`, feature `cuda`). That
build ships CUDA **12 and 13** prebuilts. On CUDA 12 hosts:

```bash
export ORT_CUDA_VERSION=12
```

## 1. Host-only tests (no GPU)

Schema builder, graph crate, and `rocky` (ONNX → schema). Safe for laptops
and CI.

```bash
cargo test -p rmlk-schema -p rmlk-graph -p rocky
```

Coverage (CI):

```bash
cargo llvm-cov -p rmlk-schema -p rmlk-graph -p rocky --lcov --output-path lcov.info
```

### E2e asset resolver (host)

Fake directory backend; no Hub credentials. Exercises override → cache →
download order and SHA-256 checks.

```bash
cargo test -p rmlk-runtime --test e2e_assets
```

Contract: [`e2e-artifacts.md`](e2e-artifacts.md). Implementation:
`runtime/tests/common/e2e_assets.rs`.

## 2. CUDA op unit tests

Per-op cases live next to the kernels in
`runtime/src/providers/cuda/backend/<op>.rs` under `#[cfg(test)]`. They use
`OpTest` / `assert_close` from `runtime/src/testing.rs` (builds a one-op graph
through `GraphBuilder` + `Builder::from_graph`).

```bash
export CUDA_COMPUTE_CAP=89   # if needed
cargo test -p rmlk-runtime --lib
```

Filter one op:

```bash
cargo test -p rmlk-runtime --lib add::
```

Typical coverage per op: happy path, supported dtypes, broadcast / scalar
shapes, unsupported-dtype errors. A few cases may be `#[ignore]` with a
reason (for example known Float16 drop issues).

## 3. Multi-op graph integration

`runtime/tests/graphs/`:

| Test | What it checks |
|------|----------------|
| `attention_head` | Small attention fragment (MatMul, Transpose, Div, Trilu, Softmax) |
| `conv_block` | Conv-style block |
| `serialize_roundtrip` | Graph serialize / load path |

Expected float blobs are checked in under `runtime/tests/fixtures/` and were
produced by the pinned CPU ORT via:

```bash
python3 scripts/gen_graph_fixtures.py
# or: .venv-oracle/bin/python scripts/gen_graph_fixtures.py
```

Run:

```bash
export CUDA_COMPUTE_CAP=89   # if needed
cargo test -p rmlk-runtime --test graphs
```

After changing graphs in `runtime/tests/graphs/`, regenerate fixtures and
commit intentional diffs. The script refuses to run if `onnx` /
`onnxruntime` versions do not match the compatibility profile.

## 4. Official ONNX node suite (GPU, ignored)

Discovers single-op cases from the pinned `onnx` package (not stored in
git). Runs cases whose ops are all in rmlk’s supported set; compares to
ONNX `.pb` oracles.

Classification:

1. **Skipped (unknown op)** — op not in rmlk’s `Op` set.
2. **`dtype_gap`** — known ops, but this dtype/cast path is unimplemented
   (informational; does not fail the suite by itself).
3. **`FAIL`** — wrong outputs or unexpected error (fails the test).
4. **Passed** — within tolerance (`atol=1e-5`, `rtol=1e-4`).

```bash
python3 scripts/discover_onnx_node_cases.py
# If the manifest is missing, the test invokes discover automatically.

export CUDA_COMPUTE_CAP=89   # if needed
cargo test -p rmlk-runtime --test onnx_node -- --ignored --nocapture
```

Cache override: `RMLK_ONNX_NODE_CACHE`. There is **no** bless/baseline file;
the printed summary is the source of truth for gaps.

Known notes (see also `compatibility.md`): some `ConstantOfShape` cases are
skipped until packed tensor attributes load; Softmax axis-1 rank-3 still
fails until that path exists.

## 5. Full-model e2e vs ORT (GPU, ignored)

Harness: `runtime/tests/e2e_ort/`. Resolves artifacts (override → cache →
Hugging Face / HTTPS), runs **pinned ORT CUDA EP** and **rmlk**, compares
outputs.

Registered cases (`runtime/tests/e2e_ort/cases.rs`):

| Case id | Kind | What is checked |
|---------|------|-----------------|
| `resnet34` | Single forward | ImageNet preprocess of versioned `dog.jpeg`; float closeness; top-1 **207** |
| `llama3.2` | Greedy decode | Short chat-templated prompt; next-token ids + last-row logits vs ORT (`max_new_tokens=8`) |

Llama runs ORT first, drops the session, then runs rmlk (teacher-forced with
ORT’s tokens) so only one ~12 GiB weight load sits on the GPU at a time.

### Pack artifacts once

Large weights stay out of git. Pack into `.cache/rmlk/e2e/` (or use path
overrides).

```bash
# ResNet34: converts ONNX to opset 14, copies dog.jpeg, writes sidecar
PYTHONPATH=.venv-oracle/lib/python3.12/site-packages \
  python3 scripts/pack_e2e_resnet34.py --model /path/to/resnet34.onnx

# Llama-3.2: symlinks model.onnx + model.onnx_data, copies tokenizer + prompt
PYTHONPATH=.venv-oracle/lib/python3.12/site-packages \
  python3 scripts/pack_e2e_llama32.py \
    --model-dir ~/Models/Llama-3.2-3B-Instruct
```

Committed fixtures used as pack sources:

- `runtime/tests/fixtures/e2e/resnet34/dog.jpeg`
- `runtime/tests/fixtures/e2e/llama3.2/prompt.txt` (`Say hi in one short sentence.`)

Artifact contract (roles, checksums, Hub keys): [`e2e-artifacts.md`](e2e-artifacts.md).

### Run the full e2e suite (ResNet + Llama)

Minimal (artifacts already in default cache from pack):

```bash
export ORT_CUDA_VERSION=12
export CUDA_COMPUTE_CAP=89
cargo test -p rmlk-runtime --test e2e_ort -- --ignored --nocapture
```

Recommended local overrides for Llama so external `model.onnx_data` resolves
beside the real ONNX file (paths from `pack_e2e_llama32.py` output; adjust
to your machine):

```bash
export ORT_CUDA_VERSION=12
export CUDA_COMPUTE_CAP=89

export RMLK_E2E_ASSET_LLAMA3_2_MODEL=~/Models/Llama-3.2-3B-Instruct/onnx/model.onnx
export RMLK_E2E_ASSET_LLAMA3_2_MODEL_DATA=~/Models/Llama-3.2-3B-Instruct/onnx/model.onnx_data
export RMLK_E2E_ASSET_LLAMA3_2_TOKENIZER=$PWD/.cache/rmlk/e2e/llama3.2/2026-09-27/tokenizer.json
export RMLK_E2E_ASSET_LLAMA3_2_PROMPT=$PWD/.cache/rmlk/e2e/llama3.2/2026-09-27/prompt.txt
export RMLK_E2E_ASSET_LLAMA3_2_SIDECAR=$PWD/.cache/rmlk/e2e/llama3.2/2026-09-27/sidecar.json

# Optional ResNet overrides if not using cache/Hub:
# export RMLK_E2E_ASSET_RESNET34_MODEL=…
# export RMLK_E2E_ASSET_RESNET34_IMAGE=…
# export RMLK_E2E_ASSET_RESNET34_SIDECAR=…

cargo test -p rmlk-runtime --test e2e_ort -- --ignored --nocapture
```

Host-only smoke for the e2e crate (case registry, compare helpers; does
**not** load models):

```bash
cargo test -p rmlk-runtime --test e2e_ort
```

### Env vars (e2e)

| Variable | Purpose |
|----------|---------|
| `ORT_CUDA_VERSION` | `12` or `13` — select ort’s CUDA prebuilt |
| `CUDA_COMPUTE_CAP` | sm version when nvidia-smi probe fails |
| `RMLK_E2E_HF_REPO` / `RMLK_E2E_HF_REVISION` | Hub repo for downloads |
| `RMLK_E2E_BASE_URL` | Explicit HTTPS prefix (overrides Hub-derived URL) |
| `HF_TOKEN` | Bearer token for private Hub repos |
| `RMLK_E2E_CACHE` | Cache root (default `<repo>/.cache/rmlk/e2e`) |
| `RMLK_E2E_ASSET_<CASE>_<ROLE>` | Local file override for one role |

Naming: uppercase, non-alnum → `_`. Example: `llama3.2` + `model` →
`RMLK_E2E_ASSET_LLAMA3_2_MODEL`.

## Suggested workflows

### Day-to-day refactor (fast)

```bash
cargo test -p rmlk-schema -p rmlk-graph -p rocky
cargo test -p rmlk-runtime --test e2e_assets
export CUDA_COMPUTE_CAP=89
cargo test -p rmlk-runtime --lib
cargo test -p rmlk-runtime --test graphs
```

### Before merging a large runtime / CUDA change

Run the day-to-day set, then:

```bash
cargo test -p rmlk-runtime --test onnx_node -- --ignored --nocapture
export ORT_CUDA_VERSION=12
cargo test -p rmlk-runtime --test e2e_ort -- --ignored --nocapture
```

Expect ResNet top-1 `207` and Llama next-token agreement through a short
greedy decode. Failures name output, worst index, and tolerance.

### After changing IR / opset / ORT pins

Follow the bump checklist in [`compatibility.md`](compatibility.md): update
pins, re-run node suite, regenerate graph fixtures, re-run full-model e2e,
refresh artifact sidecars.

## Layout cheat sheet

```text
runtime/
  src/providers/cuda/backend/   # Op unit tests (OpTest)
  src/testing.rs                # OpTest helpers
  tests/
    common/e2e_assets.rs        # Artifact resolver
    e2e_assets/                 # Resolver unit tests
    e2e_ort/                    # Full-model harness (ResNet, Llama)
    graphs/                     # Multi-op integration
    onnx_node/                  # Official ONNX node suite
    fixtures/                   # Graph goldens + e2e pack sources
scripts/
  requirements-oracle.in
  gen_graph_fixtures.py
  discover_onnx_node_cases.py
  pack_e2e_resnet34.py
  pack_e2e_llama32.py
docs/
  testing.md                    # This file
  compatibility.md
  e2e-artifacts.md
```
