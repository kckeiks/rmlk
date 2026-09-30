# Running `rmlk-server` tests

Brief guide for integration tests under `server/tests/`. Data layout,
resolve rules, and packing: [`test-data.md`](test-data.md). Types/fields:
rustdoc on [`tests/utils`](../tests/utils/mod.rs).

## Default (no model)

From `server/` (or `-p rmlk-server` from the repo root):

```bash
cargo test -p rmlk-server --test session
cargo test -p rmlk-server --test e2e
```

| Suite | What it checks |
|-------|----------------|
| **session** | Protocol / session lifecycle over WS with the **mock** engine. |
| **e2e** (default) | `normalize_transcript` edge cases only — no GPU, no WAVs. |

Library unit tests (`cargo test -p rmlk-server --lib`) cover protocol, session
registry, and HTTP helpers. Without `--features ort`, the ORT backend is not
built.

## Real ORT e2e (ignored)

Needs a Nemotron ONNX dir and packed WAVs for the clips in
`tests/manifests/e2e.json`. Streams each clip over WebSocket to the
server and requires Final text to match the manifest `reference` after
normalize.

```bash
# once: pack the small e2e set (rewrites tests/manifests/e2e.json)
python3 server/scripts/pack_librispeech.py --config clean --split test \
  --id 6930-75918-0000,6930-75918-0001,6930-75918-0002

RMLK_NEMOTRON_MODEL_DIR=/path/to/nemotron-onnx \
  cargo test -p rmlk-server --features ort --test e2e -- --ignored --nocapture
```

Optional: pack the full `clean`/`test` pool into `.cache/rmlk/testdata/`
(no `--id`) for reuse by later stress/bench jobs — see the pack script `--help`
and [`test-data.md`](test-data.md).

## Env vars (quick)

| Variable | Purpose |
|----------|---------|
| `RMLK_NEMOTRON_MODEL_DIR` | ONNX model directory |
| `RMLK_TESTDATA_CACHE` | Flat `{id}.wav` directory (default `<repo>/.cache/rmlk/testdata`) |

Details: [`test-data.md`](test-data.md).
