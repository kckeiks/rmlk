//! ASR corpus helpers (Phase 6).
//!
//! Contract: `server/docs/corpus.md`. Loads a per-corpus committed manifest +
//! gold files, resolves WAV paths (override → corpus dir → cache), and
//! compares transcripts after normalization.

#![allow(dead_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Env: flat directory of `{clip_label}.wav` files.
pub const CORPUS_DIR_ENV: &str = "RMLK_ASR_CORPUS_DIR";
/// Env: cache root (default `<repo>/.cache/rmlk/asr`).
pub const CACHE_ENV: &str = "RMLK_ASR_CACHE";

const ASR_FIXTURES_DIR: &str = "tests/fixtures/asr";

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub schema_version: u64,
    pub corpus: String,
    pub corpus_edition: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub pcm: String,
    pub chunk_samples: usize,
    pub clips: Vec<Clip>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Clip {
    pub wav: String,
    pub gold: String,
    pub sha256: String,
    pub bytes: u64,
    pub duration_s: f64,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub source_note: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
}

impl Clip {
    /// Clip label = WAV filename without the `.wav` suffix (e.g. `utt001.wav` → `utt001`).
    pub fn label(&self) -> &str {
        self.wav
            .strip_suffix(".wav")
            .unwrap_or_else(|| panic!("clip wav must end in .wav: {}", self.wav))
    }
}

/// `tests/fixtures/asr` under this crate (`CARGO_MANIFEST_DIR`).
fn asr_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(ASR_FIXTURES_DIR)
}

/// Absolute path to `{corpus}/manifest.json` under the ASR fixtures dir.
fn corpus_manifest_path(corpus: &str) -> PathBuf {
    asr_fixtures_dir().join(corpus).join("manifest.json")
}

/// Load the committed manifest for `corpus` (`correctness`, `stress`, …).
pub fn load_manifest_for(corpus: &str) -> Manifest {
    let path = corpus_manifest_path(corpus);
    let text = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!("failed to read manifest {}: {err}", path.display())
    });
    let manifest: Manifest = serde_json::from_str(&text).unwrap_or_else(|err| {
        panic!("failed to parse manifest {}: {err}", path.display())
    });
    assert_eq!(
        manifest.corpus, corpus,
        "manifest corpus field {:?} must match directory {corpus:?}",
        manifest.corpus
    );
    manifest
}

/// Load the correctness corpus manifest.
pub fn load_manifest() -> Manifest {
    load_manifest_for("correctness")
}

impl Manifest {
    /// Look up a clip by label (WAV stem, e.g. `utt001`).
    pub fn clip(&self, label: &str) -> &Clip {
        self.clips
            .iter()
            .find(|c| c.label() == label)
            .unwrap_or_else(|| panic!("clip {label} not in {} manifest", self.corpus))
    }
}

/// Path to an in-repo gold file for `clip`.
pub fn gold_path(manifest: &Manifest, clip: &Clip) -> PathBuf {
    asr_fixtures_dir()
        .join(&manifest.corpus)
        .join(&clip.gold)
}

/// Read gold transcript text (trims a single trailing newline).
pub fn load_gold(manifest: &Manifest, clip: &Clip) -> String {
    let path = gold_path(manifest, clip);
    let text = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!("failed to read gold {}: {err}", path.display())
    });
    text.trim_end_matches(['\r', '\n']).to_string()
}

fn override_env_name(clip_label: &str) -> String {
    let mut key = String::from("RMLK_ASR_UTT_");
    for ch in clip_label.chars() {
        if ch.is_ascii_alphanumeric() {
            key.push(ch.to_ascii_uppercase());
        } else {
            key.push('_');
        }
    }
    key.push_str("_WAV");
    key
}

fn cache_root(manifest: &Manifest) -> PathBuf {
    if let Ok(p) = env::var(CACHE_ENV) {
        return PathBuf::from(p)
            .join(&manifest.corpus)
            .join(&manifest.corpus_edition);
    }
    // Default cache lives at <repo>/.cache/rmlk/asr; this crate is <repo>/server.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR has a parent")
        .join(".cache/rmlk/asr")
        .join(&manifest.corpus)
        .join(&manifest.corpus_edition)
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

/// Resolve a local WAV path for `clip` (override → corpus dir → cache).
///
/// Verifies size + SHA-256 against the manifest. Panics with a clear message
/// if nothing resolves.
pub fn resolve_wav(manifest: &Manifest, clip: &Clip) -> PathBuf {
    let mut candidates: Vec<(String, PathBuf)> = Vec::new();

    let override_key = override_env_name(clip.label());
    if let Ok(p) = env::var(&override_key) {
        candidates.push((override_key, PathBuf::from(p)));
    }
    if let Ok(dir) = env::var(CORPUS_DIR_ENV) {
        candidates.push((
            CORPUS_DIR_ENV.to_string(),
            PathBuf::from(dir).join(&clip.wav),
        ));
    }
    candidates.push((
        "cache".to_string(),
        cache_root(manifest).join(&clip.wav),
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
        "could not resolve WAV for clip {} (see server/docs/corpus.md).\nTried:\n  {}",
        clip.label(),
        errors.join("\n  ")
    );
}

/// Normalize transcript text for gold compare: lowercase, strip punctuation,
/// collapse whitespace.
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

/// Compare hypothesis to gold after [`normalize_transcript`].
pub fn assert_gold_match(hypothesis: &str, gold: &str) {
    let hyp_n = normalize_transcript(hypothesis);
    let gold_n = normalize_transcript(gold);
    assert_eq!(
        hyp_n, gold_n,
        "transcript mismatch after normalize\n  hypothesis: {hypothesis:?}\n  gold:       {gold:?}\n  hyp_norm:   {hyp_n:?}\n  gold_norm:  {gold_n:?}"
    );
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}
