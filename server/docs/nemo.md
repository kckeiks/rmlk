# In-process NeMo backend

Build the server with `--features nemo` and select `--engine nemo`. The adapter
uses `nemo-speech` through its public API and loads one native recognizer.
The default server build still uses the mock backend. Existing ORT builds and
the binary WebSocket protocol are supported separately.

## Setup and launch

Install the **complete pinned NeMo ASR SDK**, including its dependent libraries,
using the [bindings setup guide](https://github.com/kckeiks/nemo-speech/blob/262354c79631f2b32a69203416d1584d1300dfaf/README.md#install-a-compatible-sdk).
The verified upstream revision is
`4c101bc7113f49101a3e11d2c994c519f41939f6`. Loading the library executes native
code: use a trusted build that implements that revision's ABI and threading
contracts. Cargo builds the Rust adapter; it does not build or download the SDK,
CUDA toolkit, driver, or model.

For GPU execution, use a CUDA SDK and an installed NVIDIA driver/CUDA runtime.
The server's `cuda` Cargo feature selects ORT CUDA; it is unnecessary for NeMo.
Choose persistent SDK/model directories. There is no required `/tmp` location
and no dependency file to manually populate. Keep the installed SDK together.

Supply an official compatible **streaming GGUF model**, rather than the ONNX
model directory used by parakeet-rs. The evaluated model was NVIDIA
Nemotron 3.5 ASR streaming 0.6B Q8_0; provenance appears below.

```sh
cargo run --release -p rmlk-server --features nemo -- \
  --engine nemo \
  --nemo-library /opt/nemo-speech/lib/libnemo_speech_asr_c.so.1 \
  --nemo-model /models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  --nemo-device 0 \
  --nemo-max-sessions 8 \
  --nemo-batch-size 8 \
  --bind 127.0.0.1:8080
```

Library paths must be absolute. `--nemo-device 0` explicitly requests GPU 0;
startup fails if the SDK cannot select it. `--nemo-device=-1` explicitly selects
CPU. GPU selection alone is not a certificate that every native operator runs
on GPU; use the pinned SDK's backend coverage tooling to audit graph placement.

All NeMo settings have environment equivalents:

- `--nemo-library`: `RMLK_NEMO_LIBRARY` (required).
- `--nemo-model`: `RMLK_NEMO_MODEL` (required).
- `--nemo-device`: `RMLK_NEMO_DEVICE` (default `0`).
- `--nemo-max-sessions`: `RMLK_NEMO_MAX_SESSIONS` (default `8`, range `1..=64`).
- `--nemo-batch-size`: `RMLK_NEMO_BATCH_SIZE` (default `8`, range `1..=max_sessions`;
  `1` disables native batching). Set this too when reducing session capacity.
- `--nemo-queue-delay-us`: `RMLK_NEMO_QUEUE_DELAY_US` (default `5000`).
- `--nemo-right-context`: `RMLK_NEMO_RIGHT_CONTEXT` (default `6`; the evaluated
  Nemotron 560 ms operating point). Other models must support the selected geometry.

`RMLK_ENGINE=nemo` and `RMLK_BIND` also work. `/health` returns `ok`; `/ws`
accepts the existing [protocol](protocol.md). Send mono PCM16LE at **16 kHz**
without WAV headers. A 560 ms frame contains 8960 samples; final audio may be
shorter. `Finalize` flushes the real audio tail and returns one wire `Final`.

## Scheduling and lifecycle

The model is loaded once. Cloned recognizers share its weights. Each admitted
session gets one worker lane backed by a long-lived Tokio blocking task; its
native stream is created, driven, and destroyed on that thread. Backend state
never crosses the async work channel. The standard Tokio runtime has enough
blocking threads for this bounded pool; custom embedding runtimes must provide
at least `max_sessions` blocking threads plus capacity for their other work.

Native calls from distinct lanes can overlap, allowing the SDK to batch
compatible frontend/encoder/decoder stages. `batch_size` is an upper bound;
actual batches depend on timing and compatibility. No global Rust inference
mutex serializes those calls. The C ABI exposes no batch metrics, so this server
test does not measure achieved physical batch widths.

Admission reserves a lane before native stream creation. Excess opens receive
protocol `BUSY` and Close. Native state-arena slots equal the maximum admitted
sessions when batching is enabled. Each lane has an 18-item FIFO queue; each
session can have at most 16 audio items queued/running, with a five-second
queue/permit timeout. Control and audio retain per-session order. These are
item limits, not byte or utterance-duration limits. The existing protocol payload
limit still applies. This setting is an admission cap, not a measured safe
realtime capacity.

The adapter converts PCM16 to float32 on CPU, then sends audio to the SDK for
frontend and neural inference. Transport, queueing, result copying, and transcript
assembly stay in Rust. Interim hypotheses replace the current utterance; native
utterance finals are retained as completed text until the connection's Finalize.

Cancel/disconnect discards queued audio and drops native state. The SDK has no
ASR abort operation: an inference call already in flight must return before
cleanup completes. Native failures close that session; an engine worker panic
is a process-level failure. The server main task monitors the aggregate worker
join handle and exits nonzero if a worker fails.

## Tests and recorded validation

Host regression checks require no SDK, model, or GPU:

```sh
cargo test -p rmlk-server
cargo test -p rmlk-server --features nemo
cargo fmt -p rmlk-server -- --check
```

On the existing Rust 1.95 baseline, strict Clippy requires allowances for two
pre-existing lint categories in protocol code and feature assertions. The adapter
and worker changes pass this check:

```sh
cargo clippy -p rmlk-server --features nemo --all-targets -- \
  -D warnings -A clippy::manual_is_multiple_of \
  -A clippy::assertions_on_constants
```

The additional worker tests cover non-Send thread-bound state, overlapping
calls, bounded admission, slot reuse after cancel/drop, and failed-open cleanup.
Existing mock WebSocket tests continue to check protocol/error/backpressure
behavior. Native transcription is an **ignored, opt-in** test; it is never
silently counted as GPU verification during ordinary Cargo tests.

For the native test, prepare the three WAVs in `tests/manifests/e2e.json` using
[test-data.md](test-data.md). The harness verifies their size and SHA-256 before
streaming. Set `RMLK_TESTDATA_CACHE` if they are outside the default cache.

```sh
RMLK_NEMO_LIBRARY=/opt/nemo-speech/lib/libnemo_speech_asr_c.so.1 \
RMLK_NEMO_MODEL=/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  cargo test -p rmlk-server --features nemo --test nemo -- \
  --ignored --nocapture
```

GPU 0 is the native test default. Add `RMLK_NEMO_DEVICE=-1` for an explicit CPU
run. The test admits four sessions, rejects a fifth, then transcribes four
simultaneous clients using 560 ms wall-clock pacing. It requires exact reference
matches after punctuation/case normalization for all three distinct recordings,
including the 14.225-second clip. It also checks cancellation, disconnect cleanup,
a fresh transcription after cleanup, and worker shutdown.

On **2026-10-01**, this test passed on the RTX 4090 using the pinned CUDA SDK,
with the runtime reporting `backend=CUDA0`. All four concurrent transcripts
matched; admission and cleanup checks passed. The feature-enabled host suite
also passed: 64 unit tests and 16 existing integration tests. The recorded GPU
run was a debug build; it is correctness evidence, not a performance benchmark.

Model provenance:

- Repository: `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- Revision: `1c8deaecc64b91f034d73e08dd8b64625eb3395d`.
- File: `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`, 741548352 bytes.
- SHA-256: `a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae`.

For sustained realtime capacity and long-session evaluation, use the
[`bench_capacity` procedure](capacity.md): continuous streams, the three-clip
mix, an independent 20 ms producer clock, JSON/Markdown reports, and GPU
sampling. For a Dirigo comparison, use the same recordings, right context,
pacing, hardware, and realtime gate. The harness measures each Audio frame
through its `AudioProcessed` acknowledgment, independently of partial hypotheses.
`AudioProcessed` means the worker has completed available work for that frame;
model lookahead can retain audio for a later decode. This backend suppresses
unchanged hypotheses, so an audio frame does not necessarily produce a partial
transcript. The reported `safe_n` requires a passing soak. Native batch-width
metrics are not exposed by the C ABI. Older finite-utterance benchmark results
need rerunning before comparing them to this workload.

## Binding dependencies

The optional `nemo-speech` dependency is pinned to Git commit
`262354c79631f2b32a69203416d1584d1300dfaf` in `server/Cargo.toml`. Its raw
`nemo-speech-sys` dependency is pinned to
`c8bf763bc9f4520f58358feaf1b79740e6f2ec9f`. Both are maintained in independent
repositories; no local binding packages or sibling checkouts are required.
The server adapter uses only public binding APIs. Cargo fetches the Rust
packages; install the native SDK separately as described above.
