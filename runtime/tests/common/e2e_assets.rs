//! Test-only resolver for full-model e2e artifacts.
//!
//! Contract: `docs/e2e-artifacts.md`. Resolution order is override → cache →
//! download via an [`AssetBackend`], then SHA-256 verify against the sidecar.

// Shared by several integration-test binaries; not every helper is used by each.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const SIDECAR_SCHEMA_VERSION: u64 = 1;
const SIDECAR_ROLE: &str = "sidecar";
const SIDECAR_FILENAME: &str = "sidecar.json";

/// Fetches object bytes by key (`{case_id}/{artifact_id}/{filename}`).
pub trait AssetBackend {
    fn fetch(&self, object_key: &str) -> Result<Vec<u8>, ResolveError>;
}

/// Local directory that mirrors the remote key layout. Used as a fake store in
/// unit tests so no network / Hub credentials are required.
pub struct DirBackend {
    root: PathBuf,
}

impl DirBackend {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl AssetBackend for DirBackend {
    fn fetch(&self, object_key: &str) -> Result<Vec<u8>, ResolveError> {
        let path = self.root.join(object_key);
        fs::read(&path).map_err(|e| ResolveError::Backend {
            key: object_key.to_string(),
            message: format!("{}: {e}", path.display()),
        })
    }
}

/// HTTPS GET against `{base_url}/{object_key}` (Hugging Face Hub resolve URL or
/// any static HTTPS prefix with the same key layout).
pub struct HttpBackend {
    base_url: String,
    /// Optional Hub token (`HF_TOKEN`); sent as Bearer auth when set.
    token: Option<String>,
}

impl HttpBackend {
    pub fn new(base_url: impl Into<String>) -> Self {
        let base = base_url.into();
        Self {
            base_url: base.trim_end_matches('/').to_string(),
            token: None,
        }
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        let t = token.into();
        if !t.is_empty() {
            self.token = Some(t);
        }
        self
    }
}

impl AssetBackend for HttpBackend {
    fn fetch(&self, object_key: &str) -> Result<Vec<u8>, ResolveError> {
        let url = format!("{}/{object_key}", self.base_url);
        let mut request = ureq::get(&url);
        if let Some(token) = &self.token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        let response = request.call().map_err(|e| ResolveError::Backend {
            key: object_key.to_string(),
            message: e.to_string(),
        })?;
        let mut bytes = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut bytes)
            .map_err(|e| ResolveError::Backend {
                key: object_key.to_string(),
                message: e.to_string(),
            })?;
        Ok(bytes)
    }
}

/// Hugging Face Hub resolve-URL prefix for a repo + revision.
pub fn huggingface_base_url(repo_id: &str, revision: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}",
        repo_id.trim_matches('/'),
        revision.trim_matches('/')
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sidecar {
    pub schema_version: u64,
    pub case_id: String,
    pub artifact_id: String,
    pub onnx_ir_version: u64,
    pub onnx_opset: u64,
    pub ort_version: String,
    pub files: HashMap<String, FileEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<i64>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("unknown role `{role}` in sidecar for {case_id}/{artifact_id}")]
    UnknownRole {
        case_id: String,
        artifact_id: String,
        role: String,
    },
    #[error("override path missing for {case_id}/{role}: {path}")]
    OverrideMissing {
        case_id: String,
        role: String,
        path: String,
    },
    #[error("checksum mismatch for {key}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        key: String,
        expected: String,
        actual: String,
    },
    #[error("size mismatch for {key}: expected {expected} bytes, got {actual}")]
    SizeMismatch {
        key: String,
        expected: u64,
        actual: u64,
    },
    #[error("sidecar schema_version {got} unsupported (want {SIDECAR_SCHEMA_VERSION})")]
    BadSidecarSchema { got: u64 },
    #[error("sidecar case_id/artifact_id mismatch: file has {got_case}/{got_art}, want {want_case}/{want_art}")]
    SidecarIdentity {
        got_case: String,
        got_art: String,
        want_case: String,
        want_art: String,
    },
    #[error("backend fetch failed for `{key}`: {message}")]
    Backend { key: String, message: String },
    #[error("no backend configured and cache miss for `{key}`")]
    NoBackend { key: String },
    #[error("{0}")]
    Io(String),
    #[error("invalid sidecar JSON: {0}")]
    SidecarJson(String),
}

pub struct AssetResolver<B: AssetBackend> {
    cache_root: PathBuf,
    backend: Option<B>,
    /// Explicit overrides for tests; checked before environment variables.
    overrides: HashMap<(String, String), PathBuf>,
    read_env_overrides: bool,
}

impl<B: AssetBackend> AssetResolver<B> {
    pub fn new(cache_root: impl Into<PathBuf>, backend: Option<B>) -> Self {
        Self {
            cache_root: cache_root.into(),
            backend,
            overrides: HashMap::new(),
            read_env_overrides: true,
        }
    }

    pub fn with_override(
        mut self,
        case_id: impl Into<String>,
        role: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Self {
        self.overrides
            .insert((case_id.into(), role.into()), path.into());
        self
    }

    pub fn read_env_overrides(mut self, enabled: bool) -> Self {
        self.read_env_overrides = enabled;
        self
    }

    /// Default cache root and optional HTTP backend from `RMLK_E2E_*` / Hub env.
    ///
    /// Remote selection: `RMLK_E2E_BASE_URL` if set, else
    /// `RMLK_E2E_HF_REPO` (+ `RMLK_E2E_HF_REVISION`, default `main`).
    pub fn from_env() -> AssetResolver<HttpBackend> {
        let cache = std::env::var_os("RMLK_E2E_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(default_cache_root);
        let token = std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty());
        let backend = remote_base_url_from_env().map(|base| {
            let mut http = HttpBackend::new(base);
            if let Some(t) = token {
                http = http.with_token(t);
            }
            http
        });
        AssetResolver::new(cache, backend)
    }

    pub fn resolve(
        &self,
        case_id: &str,
        artifact_id: &str,
        role: &str,
    ) -> Result<PathBuf, ResolveError> {
        if role == SIDECAR_ROLE {
            return self.resolve_sidecar_path(case_id, artifact_id);
        }
        let sidecar = self.load_sidecar(case_id, artifact_id)?;
        let entry = sidecar.files.get(role).ok_or_else(|| ResolveError::UnknownRole {
            case_id: case_id.to_string(),
            artifact_id: artifact_id.to_string(),
            role: role.to_string(),
        })?;
        let key = object_key(case_id, artifact_id, &entry.path);

        if let Some(path) = self.override_path(case_id, role) {
            self.verify_file(&path, entry, &key)?;
            return Ok(path);
        }

        let cache_path = self.cache_root.join(&key);
        if cache_path.is_file() {
            match self.verify_file(&cache_path, entry, &key) {
                Ok(()) => return Ok(cache_path),
                Err(ResolveError::ChecksumMismatch { .. } | ResolveError::SizeMismatch { .. }) => {
                    let _ = fs::remove_file(&cache_path);
                }
                Err(e) => return Err(e),
            }
        }

        let bytes = self.fetch_key(&key)?;
        verify_bytes(&bytes, entry, &key)?;
        atomic_write(&cache_path, &bytes)?;
        Ok(cache_path)
    }

    pub fn load_sidecar(
        &self,
        case_id: &str,
        artifact_id: &str,
    ) -> Result<Sidecar, ResolveError> {
        let path = self.resolve_sidecar_path(case_id, artifact_id)?;
        let text = fs::read_to_string(&path)
            .map_err(|e| ResolveError::Io(format!("read {}: {e}", path.display())))?;
        let sidecar: Sidecar = serde_json::from_str(&text)
            .map_err(|e| ResolveError::SidecarJson(e.to_string()))?;
        if sidecar.schema_version != SIDECAR_SCHEMA_VERSION {
            return Err(ResolveError::BadSidecarSchema {
                got: sidecar.schema_version,
            });
        }
        if sidecar.case_id != case_id || sidecar.artifact_id != artifact_id {
            return Err(ResolveError::SidecarIdentity {
                got_case: sidecar.case_id,
                got_art: sidecar.artifact_id,
                want_case: case_id.to_string(),
                want_art: artifact_id.to_string(),
            });
        }
        Ok(sidecar)
    }

    fn resolve_sidecar_path(
        &self,
        case_id: &str,
        artifact_id: &str,
    ) -> Result<PathBuf, ResolveError> {
        let key = object_key(case_id, artifact_id, SIDECAR_FILENAME);

        if let Some(path) = self.override_path(case_id, SIDECAR_ROLE) {
            if !path.is_file() {
                return Err(ResolveError::OverrideMissing {
                    case_id: case_id.to_string(),
                    role: SIDECAR_ROLE.to_string(),
                    path: path.display().to_string(),
                });
            }
            return Ok(path);
        }

        let cache_path = self.cache_root.join(&key);
        if cache_path.is_file() {
            return Ok(cache_path);
        }

        let bytes = self.fetch_key(&key)?;
        atomic_write(&cache_path, &bytes)?;
        Ok(cache_path)
    }

    fn override_path(&self, case_id: &str, role: &str) -> Option<PathBuf> {
        if let Some(path) = self.overrides.get(&(case_id.to_string(), role.to_string())) {
            return Some(path.clone());
        }
        if !self.read_env_overrides {
            return None;
        }
        let var = override_env_name(case_id, role);
        std::env::var_os(&var).map(PathBuf::from)
    }

    fn fetch_key(&self, key: &str) -> Result<Vec<u8>, ResolveError> {
        match &self.backend {
            Some(backend) => backend.fetch(key),
            None => Err(ResolveError::NoBackend {
                key: key.to_string(),
            }),
        }
    }

    fn verify_file(&self, path: &Path, entry: &FileEntry, key: &str) -> Result<(), ResolveError> {
        if !path.is_file() {
            return Err(ResolveError::OverrideMissing {
                case_id: key.to_string(),
                role: entry.path.clone(),
                path: path.display().to_string(),
            });
        }
        // Stream the hash so multi-GB external weight files do not need to fit
        // in memory (Llama `model.onnx_data` is ~12 GiB).
        verify_path(path, entry, key)
    }
}

pub fn override_env_name(case_id: &str, role: &str) -> String {
    format!(
        "RMLK_E2E_ASSET_{}_{}",
        env_token(case_id),
        env_token(role)
    )
}

fn env_token(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

pub fn object_key(case_id: &str, artifact_id: &str, filename: &str) -> String {
    format!("{case_id}/{artifact_id}/{filename}")
}

pub fn default_cache_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime crate parent")
        .join(".cache/rmlk/e2e")
}

pub fn remote_base_url_from_env() -> Option<String> {
    if let Ok(base) = std::env::var("RMLK_E2E_BASE_URL") {
        let base = base.trim().trim_end_matches('/').to_string();
        if !base.is_empty() {
            return Some(base);
        }
    }
    let repo = std::env::var("RMLK_E2E_HF_REPO").ok()?;
    let repo = repo.trim().trim_matches('/').to_string();
    if repo.is_empty() {
        return None;
    }
    let revision = std::env::var("RMLK_E2E_HF_REVISION")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "main".into());
    Some(huggingface_base_url(&repo, revision.trim()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(Sha256::digest(bytes))
}

fn hex_encode(digest: impl AsRef<[u8]>) -> String {
    let digest = digest.as_ref();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn verify_bytes(bytes: &[u8], entry: &FileEntry, key: &str) -> Result<(), ResolveError> {
    let actual_len = bytes.len() as u64;
    if actual_len != entry.bytes {
        return Err(ResolveError::SizeMismatch {
            key: key.to_string(),
            expected: entry.bytes,
            actual: actual_len,
        });
    }
    let actual = sha256_hex(bytes);
    if actual != entry.sha256 {
        return Err(ResolveError::ChecksumMismatch {
            key: key.to_string(),
            expected: entry.sha256.clone(),
            actual,
        });
    }
    Ok(())
}

fn verify_path(path: &Path, entry: &FileEntry, key: &str) -> Result<(), ResolveError> {
    let meta = fs::metadata(path)
        .map_err(|e| ResolveError::Io(format!("stat {}: {e}", path.display())))?;
    let actual_len = meta.len();
    if actual_len != entry.bytes {
        return Err(ResolveError::SizeMismatch {
            key: key.to_string(),
            expected: entry.bytes,
            actual: actual_len,
        });
    }
    let mut file = fs::File::open(path)
        .map_err(|e| ResolveError::Io(format!("open {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| ResolveError::Io(format!("read {}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = hex_encode(hasher.finalize());
    if actual != entry.sha256 {
        return Err(ResolveError::ChecksumMismatch {
            key: key.to_string(),
            expected: entry.sha256.clone(),
            actual,
        });
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), ResolveError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| ResolveError::Io(format!("mkdir {}: {e}", parent.display())))?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp)
            .map_err(|e| ResolveError::Io(format!("create {}: {e}", tmp.display())))?;
        f.write_all(bytes)
            .map_err(|e| ResolveError::Io(format!("write {}: {e}", tmp.display())))?;
        f.sync_all()
            .map_err(|e| ResolveError::Io(format!("sync {}: {e}", tmp.display())))?;
    }
    fs::rename(&tmp, path).map_err(|e| {
        ResolveError::Io(format!(
            "rename {} -> {}: {e}",
            tmp.display(),
            path.display()
        ))
    })?;
    Ok(())
}

/// Build a sidecar JSON value and SHA-256 metadata for tests / publishers.
pub fn file_entry_for_bytes(path: &str, bytes: &[u8]) -> FileEntry {
    FileEntry {
        path: path.to_string(),
        sha256: sha256_hex(bytes),
        bytes: bytes.len() as u64,
        dtype: None,
        shape: None,
    }
}
