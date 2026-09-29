# ASR test data

Layout, resolve rules, and compare policy for LibriSpeech fixtures used by
`server/tests`. Field-level notes live on `Manifest` / `Clip` in
`tests/utils/mod.rs`. How to **run** tests: [`tests.md`](tests.md).

## Layout

One shared WAV pool; thin **job manifests** in git list which utterances each
suite uses (checksums + official LibriSpeech references). WAVs stay out of git.

| Path | Role |
|------|------|
| `tests/fixtures/asr/manifests/e2e.json` | E2e transcript gate (~3–5 clips) |
| `tests/fixtures/asr/manifests/stress.json` | Later load/latency (not published) |
| `.cache/rmlk/asr/librispeech/{id}.wav` | Packed 16 kHz mono PCM16 WAVs |
| `.cache/rmlk/asr/openslr/*.tar.gz` | OpenSLR tarball cache (pack script) |

Hub mirror: [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr)
(`clean` / `test` ↔ OpenSLR `test-clean`). Pin `dataset_revision` when packing.

Packed WAV format: 16 000 Hz, **mono** (one channel), PCM signed 16-bit
little-endian. The server streams fixed steps of `CHUNK_SAMPLES` (8960 samples
= 560 ms at 16 kHz).

## Resolving a WAV

Lookup order used by `resolve_wav` in `tests/utils`:

1. `RMLK_ASR_CLIP_<ID>_WAV` (e.g. `6930-75918-0000` → `RMLK_ASR_CLIP_6930_75918_0000_WAV`)
2. `RMLK_ASR_LIBRISPEECH_DIR` or `RMLK_ASR_CORPUS_DIR` + `{id}.wav`
3. `{RMLK_ASR_CACHE}/librispeech/{id}.wav` (default `<repo>/.cache/rmlk/asr`)

Checksums must match the manifest. Missing audio is a hard error for real-engine
tests; default CI does not download WAVs.

## Compare policy

Normalize (lowercase, strip punctuation, collapse whitespace) then require
**exact equality** of hypothesis and LibriSpeech reference. Soft WER is not used
for this small clean e2e set — Nemotron on `test-clean` should match after
normalize.

## Packing

```bash
# shared pool: entire clean/test
python3 server/scripts/pack_librispeech.py --config clean --split test

# small e2e suite (rewrites manifests/e2e.json when --id is set)
python3 server/scripts/pack_librispeech.py --config clean --split test \
  --id 6930-75918-0000,6930-75918-0001,6930-75918-0002
```

See `server/scripts/pack_librispeech.py --help`. Do not change WAV bytes under an
existing checksum without updating the manifest.
