# ASR test corpora

Audio for ASR tests does **not** live in git (too bulky / regenerable). What
*does* live in git is a small checklist of which clips we use, their checksums,
and the expected transcripts. This doc is that contract.

Weights stay on `RMLK_NEMOTRON_MODEL_DIR` as before — they are not part of these
corpora.

## The pieces (and why each exists)

| Thing | What it is | Why we need it |
|-------|------------|----------------|
| **Clip** | One WAV of speech, plus the Final transcript we expect for it. The clip label is the WAV stem (`utt001.wav` → `utt001`); tests and env vars use that label. | The unit the harness runs and checks. |
| **Gold** | A tiny UTF-8 file in git (`utt001.gold`) holding that expected Final text. Lives next to that corpus’s manifest. | The regression oracle: if the engine’s Final changes, the test fails. |
| **Corpus** | A named set of clips with one job: `correctness` or `stress`. | Keeps “did we get the words right?” separate from capacity / latency / cost benches. |
| **Corpus edition** (`corpus_edition`) | An opaque string naming one edition of a corpus (e.g. `2026-09-28`). Chosen when you publish; not derived from the audio. | Names the suite snapshot (“correctness @ this edition”). Checksums verify each WAV. |
| **Checksums** | Per-clip SHA-256 and byte size in that corpus’s manifest. | Prove a local WAV matches the edition. The harness rejects a hash mismatch. |
| **Manifest** | One committed JSON file **per corpus**. Names the corpus and corpus edition, then lists each clip (WAV name, gold name, checksums, size). | WAVs are out of git; this is the checklist of which clips exist and how to verify them. |

Two corpora:

| Corpus | Job | Size | Status |
|--------|-----|------|--------|
| **correctness** | Transcript regressions before latency work | ~3–5 short English clips, under ~1 minute total | Seeded with `utt001` |
| **stress** | Capacity / concurrency / latency / cost-style pipeline benches (Phase 7–8) | Minutes of audio, many parallel sessions | Layout reserved; not published yet |

Correctness stays small on purpose. Do not grow it into a WER leaderboard set.
Stress is where load and economics measurements go later; same shape, bigger set.

## Audio format

Every clip WAV:

| Property | Value |
|----------|-------|
| Container | WAV |
| Sample rate | 16 000 Hz |
| Channels | 1 (mono) |
| Sample format | PCM signed 16-bit little-endian |

Streaming step size is still 8960 samples (560 ms) — `CHUNK_SAMPLES` in the crate.
Clips used for streaming tests should be longer than one step.

## Where files live

**In git** (one directory per corpus: manifest + golds):

```text
server/tests/fixtures/asr/{corpus}/manifest.json
server/tests/fixtures/asr/{corpus}/{clip_label}.gold
```

Examples:

```text
server/tests/fixtures/asr/correctness/manifest.json
server/tests/fixtures/asr/correctness/utt001.gold

server/tests/fixtures/asr/stress/manifest.json      # later
server/tests/fixtures/asr/stress/….gold             # later
```

`.gold` instead of `.txt` so the repo-wide `*.txt` gitignore does not hide them.
One transcript per file, UTF-8; a single trailing newline is fine.

**WAV cache** (not in git; default under the repo, overridable):

```text
.cache/rmlk/asr/{corpus}/{corpus_edition}/{clip_label}.wav
```

Example: `.cache/rmlk/asr/correctness/2026-09-28/utt001.wav`

| Path segment | Meaning |
|--------------|---------|
| `{corpus}` | `correctness` or `stress` |
| `{corpus_edition}` | Same string as `corpus_edition` in that corpus’s manifest |
| `{clip_label}.wav` | That clip’s audio (`utt001.wav`, …) |

## Manifest

Each corpus has its own file:

| Corpus | Path |
|--------|------|
| correctness | [`server/tests/fixtures/asr/correctness/manifest.json`](../tests/fixtures/asr/correctness/manifest.json) |
| stress | `server/tests/fixtures/asr/stress/manifest.json` (not published yet) |

### File fields

| Field | Meaning |
|-------|---------|
| `corpus` | Must match the parent directory name (`correctness` or `stress`) |
| `corpus_edition` | Edition id for this suite snapshot |
| `sample_rate_hz` / `channels` / `pcm` / `chunk_samples` | Expected audio / streaming knobs |
| `clips` | Array of clip entries (below) |

### Per-clip fields (`clips[]`)

| Field | Meaning |
|-------|---------|
| `wav` | Filename under `.cache/rmlk/asr/{corpus}/{corpus_edition}/`. Stem is the clip label (`utt001.wav` → `utt001`). |
| `gold` | Gold filename under `server/tests/fixtures/asr/{corpus}/` |
| `sha256` | Lowercase hex SHA-256 of the WAV bytes |
| `bytes` | Exact size (cheap check before hashing) |
| `duration_s` | Approximate length (docs / sanity) |
| `source` | Where the audio came from |

After the harness finds a WAV (override or cache), it checks size + SHA-256
against this list and rejects a mismatch.

## Golds

A gold is the Final string from a known-good local ORT / parakeet-rs run on
that clip, frozen when the clip was added. It need not match a human
LibriSpeech reference. Optional Triton / human-ref compare is Phase 6.4.

Compare: normalize (lowercase, strip punctuation, collapse whitespace), then
exact match. Add WER only if that proves too brittle.

## Finding a WAV at test time

Order:

1. Per-clip override env (below)
2. `RMLK_ASR_CORPUS_DIR/{clip_label}.wav` if that flat dir is set
3. Cache path under `{corpus}/{corpus_edition}/` if checksums match
4. Download later (optional): same path under `RMLK_ASR_BASE_URL`

Missing audio when a real-engine test needs it is a hard error. Default host
CI does not download and does not require WAVs.

| Variable | Purpose |
|----------|---------|
| `RMLK_NEMOTRON_MODEL_DIR` | Model weights (unchanged) |
| `RMLK_ASR_CACHE` | Cache root (default `<repo>/.cache/rmlk/asr`) |
| `RMLK_ASR_CORPUS_DIR` | Flat folder of `{clip_label}.wav` |
| `RMLK_ASR_UTT_<ID>_WAV` | One clip path (`utt001` → `RMLK_ASR_UTT_UTT001_WAV`) |
| `RMLK_ASR_BASE_URL` | Optional HTTPS prefix for future download |

```bash
export RMLK_NEMOTRON_MODEL_DIR=~/Models/nemotron-en/nemotron-speech-streaming-en-0.6b
export RMLK_ASR_CORPUS_DIR=~/Code/rmlk/.cache/rmlk/asr/correctness/2026-09-28
```

## Adding or changing a clip

1. Convert to the audio format above (`ffmpeg -ar 16000 -ac 1 -sample_fmt s16 …`).
2. Put the WAV in `.cache/rmlk/asr/{corpus}/{corpus_edition}/`.
3. Update that corpus’s `manifest.json` (checksum, size, duration).
4. Run the engine on the full clip; write Final text to
   `server/tests/fixtures/asr/{corpus}/{clip_label}.gold`.

**Do not replace WAV bytes under an existing `corpus_edition` and keep that
same edition id.** Nothing in git can stop you from overwriting a cache file;
what stops silent drift is the manifest checksum. If you change the audio but
leave the old edition id + checksums, machines that already cached the old
bytes still pass, and anyone who gets the new bytes fails the hash check — or
worse, you “fix” the checksums in place and old CI caches keep passing
against stale audio. So when the WAV content changes: new `corpus_edition`
value (and usually a new directory of that name), update the manifest
checksums, leave the old edition alone (or delete the old cache only after
nothing needs it).

### Current seed

`utt001` is HF `speech_samples/sample1.flac` converted to PCM16 WAV. Checksum
and gold are in `server/tests/fixtures/asr/correctness/`.

## Stress corpus (later)

Same layout: `server/tests/fixtures/asr/stress/manifest.json` plus golds beside
it, and WAVs under `.cache/rmlk/asr/stress/{corpus_edition}/`. For Phase 7–8:
longer clips, many files, N-client / soak / latency / capacity numbers. Not
used by the default correctness harness. Not published yet.

## Out of scope

- Model weight checksums
- Multilingual eval (add another correctness corpus edition later if needed)
- Checking WAVs into git
