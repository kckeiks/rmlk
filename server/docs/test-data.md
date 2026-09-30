# ASR test data

Layout, resolve rules, and compare policy for audio used by `server/tests`.
Field-level notes live on `Manifest` / `Clip` in `tests/utils/mod.rs`. How to
**run** tests: [`tests.md`](tests.md).

The Rust helpers are **source-agnostic**: a job manifest + WAV directory +
checksums. Today’s pack script fills that cache from OpenSLR / Hub LibriSpeech;
another provider could write the same layout later.

## Layout

One shared WAV pool; thin **job manifests** in git list which utterances each
suite uses (checksums + official references). WAVs stay out of git.

| Path | Role |
|------|------|
| `tests/manifests/e2e.json` | E2e transcript gate (~3–5 clips) |
| `tests/manifests/stress.json` | Later load/latency (not published) |
| `.cache/rmlk/testdata/{id}.wav` | Packed 16 kHz mono PCM16 WAVs |
| `.cache/rmlk/asr/openslr/*.tar.gz` | Upstream tarball cache (LibriSpeech pack script) |

Current e2e clips come from [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr)
(`clean` / `test`). Pin `dataset_revision` when packing.

Packed WAV format: 16 000 Hz, **mono** (one channel), PCM signed 16-bit
little-endian. The server streams fixed steps of `CHUNK_SAMPLES` (8960 samples
= 560 ms at 16 kHz).

## Resolving a WAV

`resolve_wav` loads `{dir}/{id}.wav`, where `dir` is `RMLK_TESTDATA_CACHE`
if set, else `<repo>/.cache/rmlk/testdata/` (pack-script default).

Checksums must match the manifest. Missing audio is a hard error for real-engine
tests; default CI does not download WAVs.

## Compare policy

Normalize (lowercase, strip punctuation, collapse whitespace) then require
**exact equality** of hypothesis and manifest `reference`. Soft WER is not used
for this small clean e2e set.

## Packing (LibriSpeech provider)

```bash
# shared pool: entire clean/test
python3 server/scripts/pack_librispeech.py --config clean --split test

# small e2e suite (rewrites tests/manifests/e2e.json when --id is set)
python3 server/scripts/pack_librispeech.py --config clean --split test \
  --id 6930-75918-0000,6930-75918-0001,6930-75918-0002
```

See `server/scripts/pack_librispeech.py --help`. Do not change WAV bytes under an
existing checksum without updating the manifest.
