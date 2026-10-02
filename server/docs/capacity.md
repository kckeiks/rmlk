# Continuous realtime capacity comparison

Use `bench_capacity` to compare rmlk with Dirigo's
`nemotron_triton_560` capacity profile. Previous finite-utterance benchmark
results used reconnects, acknowledgment-paced chunks, and finalization, which
exercise a different workload. Rerun both backends with the matched procedure
below before comparing capacity numbers.

## Run rmlk

Generate the pinned synthetic pack if it is missing (requires Python + NumPy):

```sh
cd /home/blackcat/Code/rmlk
python3 server/scripts/gen_capacity_load.py
cargo build --release -p rmlk-server --example bench_capacity
```

The default cache is `.cache/rmlk/testdata`; `RMLK_TESTDATA_CACHE` overrides it.
The harness checks the WAV byte sizes and SHA-256 against
`server/tests/manifests/capacity.json`. It does not download models or start an
inference server.

Start the NeMo server in a separate terminal, using your installed SDK/model:

```sh
cd /home/blackcat/Code/rmlk
cargo run --release -p rmlk-server --features nemo -- \
  --engine nemo \
  --nemo-library /tmp/rmlk-nemo-cuda-sdk/lib/libnemo_speech_asr_c.so.1 \
  --nemo-model /tmp/rmlk-nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  --nemo-device 0 \
  --nemo-max-sessions 32 \
  --nemo-batch-size 8 \
  --nemo-queue-delay-us 5000 \
  --nemo-right-context 6 \
  --bind 127.0.0.1:8080
```

These `/tmp` paths match the initial experiment; change them if you installed
the SDK/model elsewhere. Keep the admission limit at or above the benchmark's
`--max-streams`. The harness cannot read server settings through `/health`;
`--notes` records operator-supplied settings, not verified configuration.

Run the comparison and save its summary (about 18 minutes if all 32 levels pass):

```sh
cd /home/blackcat/Code/rmlk
./target/release/examples/bench_capacity \
  --url ws://127.0.0.1:8080/ws \
  --start-streams 1 --max-streams 32 --ramp-step 1 \
  --warmup-seconds 8 --hold-seconds 30 --soak-seconds 120 \
  --frame-ms 20 --server-chunk-ms 560 \
  --min-realtime-ratio 0.95 \
  --max-p99-client-e2e-ms 500 --max-late-frame-ratio 0.05 \
  --gpu-index 0 --gpu-sample-interval 1 \
  --notes 'nemo; Nemotron 3.5 streaming 0.6B Q8_0; CUDA0; right_context=6; max_sessions=32; batch_size=8; queue_delay_us=5000' \
  --output-dir .cache/rmlk/benchmarks/nemo-b8-matched
```

Each invocation requires a **new** output directory. Omit `--output-dir` to
create a unique timestamped directory automatically. Reports are:

- `summary.md`: readable per-level results, sustained capacity, GPU mean/peak.
- `summary.json`: configuration, clip hashes, per-stream results and GPU summaries.
- `config.json`: input provenance, start timestamp, Git revision and dirty flag.
- `levels.jsonl`: flushed after each completed warmup/ramp/soak phase.
- `gpu.csv`: timestamped device-wide utilization, memory, and power samples.

Ctrl-C writes a summary with `status=interrupted`, preserving completed levels;
the in-progress level is not claimed as completed. Exit status is `0` only for
a complete run with a passing soak, `2` for interrupted/failed capacity runs,
and nonzero for setup/I/O errors. A failed soak leaves `safe_n=null`; test a
lower level explicitly with `--start-streams N --max-streams N` to establish it.
If N=32 passes, it is a lower bound at the configured ceiling, not proof that
32 is the maximum. Repeat runs to check variability.

## Matching Dirigo

Use the same GPU, idle background workload, driver, context and model family.
Run the two servers **separately**, not simultaneously. In Dirigo's benchmarks
directory, with its Triton server already configured for right context 6:

```sh
cd /home/blackcat/Code/dirigo/benchmarks
uv run dirigo-bench-stt-capacity \
  --profile nemotron_triton_560 \
  --pack capacity_load \
  --start-streams 1 --max-streams 32 --ramp-step 1 \
  --warmup-seconds 8 --hold-seconds 30 --soak-seconds 120 \
  --frame-ms 20 --server-chunk-ms 560 \
  --min-realtime-ratio 0.95 \
  --max-p99-queue-ms 500 --max-p99-client-e2e-ms 500 \
  --max-late-frame-ratio 0.05 --gpu-sample-interval 1 \
  --notes 'Triton; record exact weights, dtype, context, batch policy and server revision'
```

The rmlk defaults match the local Dirigo implementation's:

- Alphabetically sorted, round-robin clip assignment:
  `loop_speechish.wav`, `loop_tone_a.wav`, `loop_tone_b.wav`.
- Mono 16 kHz PCM16 input and its float32 `/32768`, then PCM16 `*32767`
  truncation conversion. WAVs are loaded and converted before measurement.
- Continuous clip wrapping without zero padding; one open stream per client
  throughout each phase. Every phase opens fresh streams.
- A 20 ms producer clock independent of inference, buffering 560 ms chunks.
  One sequential request per stream; subsequent chunks queue at the client.
- E2E from full-chunk enqueue to acknowledgment, including client backlog,
  transport and server work. Linear-interpolated p50/p95/p99 over chunk samples.
- Clock origin before open; duration starts after open. Realtime ratio is audio
  duration divided by elapsed time through drain/close, averaged over streams
  that emitted frames. The sender-only ratio is also reported.
- Late frames counted when the target time has already passed **before** sleep,
  following Dirigo's clock convention. Sleep wakeup jitter is not separately
  counted. This is a producer deadline metric, not an inference deadline metric.
- Eight-second single-stream warmup excluded from capacity scoring; 30 seconds
  per ramp level and 120-second soak at the highest passing ramp level.
- RT ratio >= .95, E2E p99 <= 500 ms, late frame fraction <= .05.

The client discards the final incomplete inference chunk, reporting its sample
count, as Dirigo does. After draining it sends `Cancel` and waits for Close to
release rmlk state, without `Finalize`/EOS work. The combined drain/close limit
is two seconds. Queue overflow, open/RPC/close failures and unacknowledged chunks
always fail a level; their samples are not silently discarded. The queue is
bounded at 256 waiting chunks per client by default. Dirigo's client queue is
unbounded and its close path allows longer cleanup; this difference should only
affect overloaded/failed levels and is recorded explicitly.

## Interpretation and remaining differences

`AudioProcessed` means the worker finished the available work for that chunk.
Model lookahead may retain audio for later decoding. It does not mean that all
samples have already become text. The benchmark does not use partial transcript
events as completion markers and does not measure audio-to-word latency.

The rmlk protocol exposes no separate server queue/inference timings; those
fields are `null`, not inferred zeros. Dirigo can report those timers as well.
Compare **client E2E**, realtime ratio and passing soak N. With both latency
thresholds at 500 ms, E2E also covers the queue component. gRPC/Triton and rmlk's
WebSocket transport remain different parts of the systems being compared.

NeMo's GGUF Q8_0 model and Triton's PyTorch weights/dtype can differ numerically;
matching audio and geometry establishes a **system capacity comparison**, not
an isolated runtime comparison with identical arithmetic. Record model/SDK/server
revisions, quantization, device, right context, admission and batch settings.
For a B=1 comparison, restart rmlk with `--nemo-batch-size 1` and save a separate
report; then compare the production batch=8 setting separately. Configured batch
size does not establish achieved batch width.

GPU samples cover the entire selected device, including desktop/other processes.
The report shows sampled means/peaks by phase and stream count; these are not
per-process kernel occupancy measurements. Failed/unavailable samples remain
unavailable and are recorded as warnings. Use `--no-gpu-sampling` for host-only
tests, or `--gpu-index` to select a different local device. For a remote server,
collect telemetry on the server host; this harness samples its own machine.

Synthetic sweeps/noise do not validate speech accuracy or representative decoder
load. Follow the synthetic comparison with **the same real-speech pack** in both
harnesses. `--data-dir PATH` accepts a flat directory of mono 16 kHz PCM16 WAVs
or raw little-endian `.pcm` files. They are sorted and hashed in the report.
rmlk rejects other sample rates/channel layouts rather than silently resampling;
prepare identical audio for both systems. Dirigo's data-root override expects
its pack directory structure, so check its resolved pack paths and hashes too.

## Host verification

```sh
cargo test -p rmlk-server --example bench_capacity
cargo test -p rmlk-server --features nemo
cargo fmt -p rmlk-server -- --check
```

The example tests use a local **mock** WebSocket server. They check that delayed
acknowledgments build visible backlog without throttling the audio clock, that
errors/missing samples cannot pass, and that a short ramp/soak writes reports and
cleans up sessions. These tests do not establish GPU throughput.
