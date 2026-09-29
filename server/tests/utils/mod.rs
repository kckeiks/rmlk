//! Shared helpers for `rmlk-server` integration tests.
//!
//! - [`TestServer`] / [`WsClient`] — WebSocket harness
//! - ASR fixtures — load job manifests, resolve WAVs, normalize transcripts
//!
//! Run suites: `server/docs/tests.md`. Fixture layout / resolve / compare:
//! `server/docs/test-data.md`.

#![allow(dead_code)]

use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use rmlk_server::http::{serve_with_state, AppState};
use rmlk_server::protocol::{ClientFrame, ServerFrame};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::WebSocketStream;

// --- WebSocket harness -------------------------------------------------------

type WsStream = WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Running test server plus a connected WebSocket client.
pub struct TestServer {
    pub addr: SocketAddr,
    pub state: AppState,
    shutdown_tx: oneshot::Sender<()>,
    server: tokio::task::JoinHandle<()>,
}

impl TestServer {
    /// Bind on an ephemeral port and serve `state`.
    pub async fn spawn(state: AppState) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let serve_state = state.clone();
        let server = tokio::spawn(async move {
            let _ = serve_with_state(
                listener,
                async {
                    let _ = shutdown_rx.await;
                },
                serve_state,
            )
            .await;
        });
        Self {
            addr,
            state,
            shutdown_tx,
            server,
        }
    }

    /// Connect a WebSocket client to `/ws`.
    pub async fn connect(&self) -> WsClient {
        let (ws, _) = tokio_tungstenite::connect_async(format!("ws://{}/ws", self.addr))
            .await
            .unwrap();
        WsClient { ws }
    }

    /// Signal shutdown and join the server task.
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
        let _ = self.server.await;
    }
}

/// Thin WebSocket client that speaks the server binary protocol.
pub struct WsClient {
    ws: WsStream,
}

impl WsClient {
    /// Send a protocol client frame as one binary WebSocket message.
    pub async fn send_frame(&mut self, frame: &ClientFrame) {
        let bytes = frame.encode().expect("encode client frame");
        self.ws
            .send(WsMessage::Binary(bytes.into()))
            .await
            .expect("send ws binary");
    }

    /// Send raw binary bytes (for garbage / malformed tests).
    pub async fn send_binary(&mut self, bytes: Vec<u8>) {
        self.ws
            .send(WsMessage::Binary(bytes.into()))
            .await
            .expect("send ws binary");
    }

    /// Receive and decode the next binary server frame.
    pub async fn recv_frame(&mut self) -> ServerFrame {
        let msg = self.ws.next().await.expect("ws stream").expect("ws message");
        let WsMessage::Binary(bytes) = msg else {
            panic!("expected binary server frame, got {msg:?}");
        };
        ServerFrame::decode(&bytes).expect("decode server frame")
    }

    /// Receive the next WebSocket message (e.g. Close).
    pub async fn recv_raw(&mut self) -> WsMessage {
        self.ws.next().await.expect("ws stream").expect("ws message")
    }

    /// `Open` then expect `OpenAck`.
    pub async fn open_session(&mut self) -> u64 {
        self.send_frame(&ClientFrame::Open).await;
        match self.recv_frame().await {
            ServerFrame::OpenAck { session_id } => session_id,
            other => panic!("expected OpenAck, got {other:?}"),
        }
    }

    /// Send a WebSocket Close and drop the client.
    pub async fn close(mut self) {
        self.ws.close(None).await.expect("ws close");
    }
}

// --- ASR fixtures ------------------------------------------------------------

/// Flat directory of `{utterance_id}.wav` files (`RMLK_ASR_LIBRISPEECH_DIR`).
///
/// Prefer this when WAVs are not under the default cache tree. See
/// `server/docs/test-data.md`.
pub const LIBRISPEECH_DIR_ENV: &str = "RMLK_ASR_LIBRISPEECH_DIR";
/// Compat alias for [`LIBRISPEECH_DIR_ENV`] (`RMLK_ASR_CORPUS_DIR`).
pub const CORPUS_DIR_ENV: &str = "RMLK_ASR_CORPUS_DIR";
/// ASR cache root (`RMLK_ASR_CACHE`; default `<repo>/.cache/rmlk/asr`).
///
/// Packed WAVs live under `{cache}/librispeech/`. Model weights use
/// `RMLK_NEMOTRON_MODEL_DIR`, not this variable.
pub const CACHE_ENV: &str = "RMLK_ASR_CACHE";

const ASR_FIXTURES_DIR: &str = "tests/fixtures/asr";
const MANIFESTS_SUBDIR: &str = "manifests";
const LIBRISPEECH_CACHE_SUBDIR: &str = "librispeech";

/// Committed job manifest listing which LibriSpeech clips a test suite uses.
///
/// Lives at `tests/fixtures/asr/manifests/{job}.json`. The [`Self::job`] field
/// must match the filename stem (`e2e.json` → `"e2e"`).
///
/// Regenerated by `pack_librispeech.py` when packing with `--id` (writes
/// `e2e.json` by default). Do not hand-edit checksums without re-packing — see
/// `server/docs/test-data.md`.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    /// JSON schema version for this file.
    ///
    /// `2` is the job-manifest layout (`job` + Hub dataset fields + inlined
    /// `reference` per clip). `1` was the old per-clip `.gold` file layout and
    /// is no longer supported.
    pub schema_version: u64,
    /// Which suite this list belongs to (`e2e`, later `stress` / `bench`).
    ///
    /// Must equal the filename stem so `load_manifest_for("e2e")` cannot
    /// accidentally load a mismatched file.
    pub job: String,
    /// Hugging Face dataset id the clips were drawn from
    /// (e.g. `openslr/librispeech_asr`). Recorded for provenance; packing uses
    /// the matching OpenSLR tarball.
    pub dataset: String,
    /// Hub config name — `clean` vs `other` (noise). E2e uses `clean`.
    pub dataset_config: String,
    /// Hub split name — e2e uses `test` (OpenSLR `test-clean`), not `train`.
    pub dataset_split: String,
    /// Hub commit/revision string recorded when the clips were packed, so
    /// everyone can see which upstream pin produced the checksums.
    pub dataset_revision: String,
    /// Sample rate the packed WAVs must have, in Hz.
    ///
    /// Always `16000`: the Nemotron / Parakeet path expects 16 kHz audio, and
    /// the pack script resamples to that.
    pub sample_rate_hz: u32,
    /// Number of audio channels in each packed WAV.
    ///
    /// Always `1` (mono). A “channel” is one independent sample stream (mono =
    /// one mic; stereo = left+right). The ASR model takes a single mono stream;
    /// stereo would be mixed down at pack time.
    pub channels: u16,
    /// PCM encoding label for the packed WAVs (`s16le` = signed 16-bit
    /// little-endian integers). Matches what `hound` reads and what the engine
    /// converts to `f32`.
    pub pcm: String,
    /// Samples per streaming step the server feeds the engine.
    ///
    /// `8960` at 16 kHz is 560 ms — the same `CHUNK_SAMPLES` constant as the
    /// crate. Kept in the manifest so fixtures stay aligned if that knob moves.
    pub chunk_samples: usize,
    /// Utterances included in this job, in pack order.
    pub clips: Vec<Clip>,
    /// Optional human notes from the pack script (not read by tests).
    #[serde(default)]
    pub notes: Option<String>,
}

/// One LibriSpeech utterance listed in a [`Manifest`].
#[derive(Debug, Clone, Deserialize)]
pub struct Clip {
    /// LibriSpeech utterance id (`{speaker}-{chapter}-{utt}`), e.g.
    /// `6930-75918-0000`. Stable key across Hub and OpenSLR.
    pub id: String,
    /// Filename under the LibriSpeech cache directory (`{id}.wav`).
    pub wav: String,
    /// Ground-truth transcript from LibriSpeech (`*.trans.txt` / Hub `text`).
    ///
    /// Compared to model output after [`normalize_transcript`] — not a
    /// separate `.gold` file.
    pub reference: String,
    /// SHA-256 of the packed WAV bytes (lowercase hex). Proves the local file
    /// matches what was committed in this manifest entry.
    pub sha256: String,
    /// Exact size of the packed WAV in bytes (fast reject before hashing).
    pub bytes: u64,
    /// Approximate duration in seconds (informational; not used for gating).
    pub duration_s: f64,
}

fn asr_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(ASR_FIXTURES_DIR)
}

fn job_manifest_path(job: &str) -> PathBuf {
    asr_fixtures_dir()
        .join(MANIFESTS_SUBDIR)
        .join(format!("{job}.json"))
}

/// Load the committed manifest for `job` (`e2e`, `stress`, …).
pub fn load_manifest_for(job: &str) -> Manifest {
    let path = job_manifest_path(job);
    let text = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!("failed to read manifest {}: {err}", path.display())
    });
    let manifest: Manifest = serde_json::from_str(&text).unwrap_or_else(|err| {
        panic!("failed to parse manifest {}: {err}", path.display())
    });
    assert_eq!(
        manifest.job, job,
        "manifest job field {:?} must match filename stem {job:?}",
        manifest.job
    );
    manifest
}

/// Load the e2e job manifest.
pub fn load_manifest() -> Manifest {
    load_manifest_for("e2e")
}

impl Manifest {
    /// Look up a clip by LibriSpeech utterance id.
    pub fn clip(&self, id: &str) -> &Clip {
        self.clips
            .iter()
            .find(|c| c.id == id)
            .unwrap_or_else(|| panic!("clip {id} not in {} manifest", self.job))
    }
}

fn override_env_name(clip_id: &str) -> String {
    let mut key = String::from("RMLK_ASR_CLIP_");
    for ch in clip_id.chars() {
        if ch.is_ascii_alphanumeric() {
            key.push(ch.to_ascii_uppercase());
        } else {
            key.push('_');
        }
    }
    key.push_str("_WAV");
    key
}

fn librispeech_cache_dir() -> PathBuf {
    if let Ok(dir) = env::var(LIBRISPEECH_DIR_ENV).or_else(|_| env::var(CORPUS_DIR_ENV)) {
        return PathBuf::from(dir);
    }
    let cache_root = env::var(CACHE_ENV).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("CARGO_MANIFEST_DIR has a parent")
            .join(".cache/rmlk/asr")
            .display()
            .to_string()
    });
    PathBuf::from(cache_root).join(LIBRISPEECH_CACHE_SUBDIR)
}

/// Verify a local WAV against the manifest entry (size + SHA-256).
pub fn verify_wav(path: &Path, clip: &Clip) -> Result<(), String> {
    let meta = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = meta.len();
    if len != clip.bytes {
        return Err(format!(
            "{}: size mismatch: expected {} bytes, got {len}",
            path.display(),
            clip.bytes
        ));
    }
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let actual = sha256_hex(&bytes);
    if actual != clip.sha256 {
        return Err(format!(
            "{}: sha256 mismatch: expected {}, got {actual}",
            path.display(),
            clip.sha256
        ));
    }
    Ok(())
}

/// Resolve a local WAV path for `clip` (override → librispeech dir → cache).
///
/// Lookup order is documented in `server/docs/test-data.md`. Panics if no
/// candidate exists or checksums fail.
pub fn resolve_wav(_manifest: &Manifest, clip: &Clip) -> PathBuf {
    let mut candidates: Vec<(String, PathBuf)> = Vec::new();

    let override_key = override_env_name(&clip.id);
    if let Ok(p) = env::var(&override_key) {
        candidates.push((override_key, PathBuf::from(p)));
    }
    candidates.push((
        "librispeech_cache".to_string(),
        librispeech_cache_dir().join(&clip.wav),
    ));

    let mut errors = Vec::new();
    for (source, path) in &candidates {
        if !path.is_file() {
            errors.push(format!("{source}: not a file ({})", path.display()));
            continue;
        }
        match verify_wav(path, clip) {
            Ok(()) => return path.clone(),
            Err(err) => errors.push(format!("{source}: {err}")),
        }
    }

    panic!(
        "could not resolve WAV for clip {} (see server/docs/test-data.md).\nTried:\n  {}",
        clip.id,
        errors.join("\n  ")
    );
}

/// Normalize transcript text: lowercase, strip punctuation, collapse whitespace.
///
/// Used so LibriSpeech ALL-CAPS references compare equal to model output that
/// differs only in case or punctuation.
pub fn normalize_transcript(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_space = true;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_space = false;
        } else if ch.is_whitespace() || is_punct(ch) {
            if !last_space && !out.is_empty() {
                out.push(' ');
                last_space = true;
            }
        } else {
            for c in ch.to_lowercase() {
                if c.is_whitespace() {
                    if !last_space && !out.is_empty() {
                        out.push(' ');
                        last_space = true;
                    }
                } else {
                    out.push(c);
                    last_space = false;
                }
            }
        }
    }
    out.trim_end().to_string()
}

fn is_punct(ch: char) -> bool {
    matches!(
        ch,
        '.' | ',' | ';' | ':' | '!' | '?' | '"' | '\'' | '`' | '(' | ')' | '[' | ']' | '{'
            | '}' | '-' | '—' | '–' | '/' | '\\' | '*'
    )
}

/// Assert hypothesis matches the reference after [`normalize_transcript`].
pub fn assert_normalized_equal(hypothesis: &str, reference: &str) {
    let hyp = normalize_transcript(hypothesis);
    let refer = normalize_transcript(reference);
    assert_eq!(
        hyp, refer,
        "normalized transcripts differ\n  hypothesis: {hypothesis:?}\n  reference:  {reference:?}\n  hyp_norm:   {hyp:?}\n  ref_norm:   {refer:?}"
    );
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Read a packed 16 kHz mono PCM16 WAV into samples.
pub fn load_wav_pcm16(path: &Path) -> Vec<i16> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|err| {
        panic!("failed to open WAV {}: {err}", path.display())
    });
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "WAV must be mono");
    assert_eq!(spec.sample_rate, 16_000, "WAV must be 16 kHz");
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    assert_eq!(spec.bits_per_sample, 16);
    reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|err| panic!("failed to read WAV samples: {err}"))
}

/// Split PCM into fixed-size chunks, zero-padding the last chunk.
pub fn pcm_chunks(pcm: &[i16], chunk_samples: usize) -> Vec<Vec<i16>> {
    pcm.chunks(chunk_samples)
        .map(|chunk| {
            let mut padded = chunk.to_vec();
            if padded.len() < chunk_samples {
                padded.resize(chunk_samples, 0);
            }
            padded
        })
        .collect()
}
