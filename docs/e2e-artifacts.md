# Full-model e2e artifacts

Contract for large ONNX models and inputs used by the manual ORT-aligned e2e
suite (TODO section 14). These files do **not** live in git. The resolver
(item 58) and harness (items 59–61) follow this document.

Compatibility pins (IR, opset, ORT version) live in
[`compatibility.md`](compatibility.md). Sidecars must record the same values
that were current when the artifact was published.

## Remote store

| Field | Value |
|-------|-------|
| Scheme | HTTPS object storage (S3-compatible or a static HTTPS prefix) |
| Base URL | Set by `RMLK_E2E_BASE_URL` (no trailing slash). Example: `https://example-bucket.s3.amazonaws.com/rmlk-e2e` |
| Auth | Public read, or credentials supplied by the environment of the runner. The contract does not require a specific cloud SDK; a simple HTTP GET of `{base}/{key}` is enough. |

Until a real bucket is published, local path overrides (below) are the supported
way to run e2e. The base URL is required only when a file is not overridden and
not already present in the cache.

## Object key layout

Keys are POSIX-style paths under the base URL:

```text
{case_id}/{artifact_id}/{filename}
```

| Segment | Meaning |
|---------|---------|
| `case_id` | Stable case name used by the harness: `resnet34`, `llama3.2`, … |
| `artifact_id` | Content revision of that case's published set, typically a short content hash or dated tag (for example `2026-09-27` or `sha256:abcd1234` truncated). Changing any payload byte requires a new `artifact_id`. |
| `filename` | One of the roles listed below. |

Sidecar for a published set:

```text
{case_id}/{artifact_id}/sidecar.json
```

Example keys:

```text
resnet34/2026-09-27/model.onnx
resnet34/2026-09-27/input_labrador.f32
resnet34/2026-09-27/sidecar.json
llama3.2/2026-09-27/model.onnx
llama3.2/2026-09-27/tokenizer.json
llama3.2/2026-09-27/prompt.txt
llama3.2/2026-09-27/sidecar.json
```

Which `artifact_id` is current for a case is recorded in the committed case
manifest that the harness loads (added with items 59–61), not in this doc.

## File roles

Logical roles the resolver accepts. Filenames in the store match the role name
unless the sidecar overrides them via `files`.

| Role | Typical filename | Notes |
|------|------------------|-------|
| `model` | `model.onnx` | ONNX graph loaded by rmlk and ORT. |
| `input` | case-specific (e.g. `input_labrador.f32`) | Fixed preprocessed tensor bytes for ResNet-style cases. |
| `tokenizer` | `tokenizer.json` | Llama-style cases. |
| `prompt` | `prompt.txt` | UTF-8 prompt for greedy decode cases. |
| `sidecar` | `sidecar.json` | Metadata and checksums for the set. |

## Checksums

Every payload file (everything except the sidecar itself) has a checksum entry
in the sidecar. Algorithm: **SHA-256**, hex-encoded lowercase, field name
`sha256`.

After download (or when validating a cache hit), the resolver hashes the local
bytes and rejects the file if they do not match. Overrides are also verified
when the sidecar is available; if an override is used without a sidecar, the
harness may skip the check but should log that fact.

## Sidecar schema

`sidecar.json` is JSON, UTF-8, one object:

```json
{
  "schema_version": 1,
  "case_id": "resnet34",
  "artifact_id": "2026-09-27",
  "onnx_ir_version": 10,
  "onnx_opset": 14,
  "ort_version": "1.28.0",
  "files": {
    "model": {
      "path": "model.onnx",
      "sha256": "…64 hex chars…",
      "bytes": 87342112
    },
    "input": {
      "path": "input_labrador.f32",
      "sha256": "…",
      "bytes": 602112,
      "dtype": "f32",
      "shape": [1, 3, 224, 224]
    }
  },
  "notes": "optional free text"
}
```

| Field | Required | Meaning |
|-------|----------|---------|
| `schema_version` | yes | Sidecar format version; currently `1`. |
| `case_id` | yes | Must match the key prefix and the harness case id. |
| `artifact_id` | yes | Must match the key segment. |
| `onnx_ir_version` | yes | IR of the published model; must match [`compatibility.md`](compatibility.md) when claimed as in-profile. |
| `onnx_opset` | yes | `ai.onnx` opset of the published model. |
| `ort_version` | yes | ORT version used when validating or generating companion data; must match the compatibility pin. |
| `files` | yes | Map of role → file descriptor. |
| `files.*.path` | yes | Filename under `{case_id}/{artifact_id}/`. |
| `files.*.sha256` | yes | SHA-256 of the file bytes. |
| `files.*.bytes` | yes | Exact size; used as a cheap pre-check before hashing. |
| `files.*.dtype` / `shape` | no | For tensor blobs (inputs); helps the harness load without guessing. |
| `notes` | no | Human-readable provenance. |

Roles present in `files` are the only roles the resolver will fetch for that
artifact set.

## Local cache

Default cache root:

```text
.cache/rmlk/e2e/
```

(relative to the repository root; overridable). Layout mirrors object keys:

```text
.cache/rmlk/e2e/{case_id}/{artifact_id}/{filename}
```

A file is a cache hit only if it exists, its length matches `files.*.bytes`,
and its SHA-256 matches `files.*.sha256`. Failed verification deletes or
ignores the bad file and re-downloads.

`.cache/` remains gitignored (same as the ONNX node-suite cache).

## Resolution order

Given `(case_id, artifact_id, role)`, the resolver returns a local filesystem
path by trying, in order:

1. **Override** — environment variable for that case and role (see below).
2. **Cache** — `{cache_root}/{case_id}/{artifact_id}/{filename}` if checksums pass.
3. **Download** — `GET {base_url}/{case_id}/{artifact_id}/{filename}`, write into
   the cache atomically, verify checksum, then return the cache path.

Missing base URL when a download is required is an error. Missing override and
missing remote for a required role is an error.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `RMLK_E2E_BASE_URL` | Remote prefix for downloads (no trailing slash). |
| `RMLK_E2E_CACHE` | Override cache root (default: `<repo>/.cache/rmlk/e2e`). |
| `RMLK_E2E_ASSET_<CASE>_<ROLE>` | Absolute or repo-relative path to a local file that replaces remote+cache for that role. `CASE` and `ROLE` are uppercase, non-alnum → `_`. |

Examples:

```bash
# Use a model already on disk; skip download for that role only.
export RMLK_E2E_ASSET_RESNET34_MODEL=/data/models/resnet34.onnx
export RMLK_E2E_ASSET_RESNET34_INPUT=/data/fixtures/labrador.f32

export RMLK_E2E_ASSET_LLAMA3_2_MODEL=/data/models/llama3.2.onnx
export RMLK_E2E_ASSET_LLAMA3_2_TOKENIZER=/data/models/tokenizer.json
export RMLK_E2E_ASSET_LLAMA3_2_PROMPT=/data/fixtures/prompt.txt

# Optional: custom cache or bucket.
export RMLK_E2E_CACHE=/var/tmp/rmlk-e2e-cache
export RMLK_E2E_BASE_URL=https://example-bucket.s3.amazonaws.com/rmlk-e2e
```

Naming rule: take `case_id` and `role`, uppercase, replace every character
outside `[A-Z0-9]` with `_`, then
`RMLK_E2E_ASSET_{CASE}_{ROLE}`. So `llama3.2` + `model` →
`RMLK_E2E_ASSET_LLAMA3_2_MODEL`.

## Publishing a new artifact set

1. Export or convert the model to the IR/opset in [`compatibility.md`](compatibility.md).
2. Choose a new `artifact_id` (never overwrite bytes under an existing id).
3. Upload payload files to `{case_id}/{artifact_id}/…`.
4. Write `sidecar.json` with checksums and pin versions; upload it last.
5. Point the committed harness case manifest at the new `artifact_id`.
6. Re-run e2e; re-bless the e2e results baseline (item 64) if verdicts change
   intentionally.

## Out of scope

- Op unit tests and small graph fixtures under `runtime/tests/fixtures/` — those
  stay in git with in-repo expected values.
- Official ONNX node-suite files — discovered from the pinned `onnx` package;
  see [`compatibility.md`](compatibility.md).
