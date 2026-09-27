//! Unit tests for the e2e asset resolver (fake directory backend; no network).
//!
//! ```text
//! cargo test -p rmlk-runtime --test e2e_assets
//! ```

#[path = "../common/mod.rs"]
mod common;

use common::e2e_assets::{
    file_entry_for_bytes, huggingface_base_url, object_key, override_env_name, AssetResolver,
    DirBackend, FileEntry, HttpBackend, ResolveError, Sidecar,
};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

fn write_bytes(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, bytes).unwrap();
}

fn sample_sidecar(case_id: &str, artifact_id: &str, files: HashMap<String, FileEntry>) -> String {
    let value = serde_json::json!({
        "schema_version": 1,
        "case_id": case_id,
        "artifact_id": artifact_id,
        "onnx_ir_version": 10,
        "onnx_opset": 14,
        "ort_version": "1.28.0",
        "files": files,
    });
    serde_json::to_string_pretty(&value).unwrap() + "\n"
}

struct Fixture {
    remote: PathBuf,
    cache: PathBuf,
    case_id: String,
    artifact_id: String,
    model_bytes: Vec<u8>,
    input_bytes: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap().keep();
        let remote = root.join("remote");
        let cache = root.join("cache");
        let case_id = "resnet34".to_string();
        let artifact_id = "2026-09-27".to_string();
        let model_bytes = b"fake-onnx-bytes".to_vec();
        let input_bytes = b"fake-input-f32".to_vec();

        let mut files = HashMap::new();
        files.insert(
            "model".into(),
            file_entry_for_bytes("model.onnx", &model_bytes),
        );
        files.insert(
            "input".into(),
            file_entry_for_bytes("input_labrador.f32", &input_bytes),
        );
        let sidecar = sample_sidecar(&case_id, &artifact_id, files);

        write_bytes(
            &remote.join(object_key(&case_id, &artifact_id, "model.onnx")),
            &model_bytes,
        );
        write_bytes(
            &remote.join(object_key(&case_id, &artifact_id, "input_labrador.f32")),
            &input_bytes,
        );
        write_bytes(
            &remote.join(object_key(&case_id, &artifact_id, "sidecar.json")),
            sidecar.as_bytes(),
        );

        Self {
            remote,
            cache,
            case_id,
            artifact_id,
            model_bytes,
            input_bytes,
        }
    }

    fn resolver(&self) -> AssetResolver<DirBackend> {
        AssetResolver::new(self.cache.clone(), Some(DirBackend::new(&self.remote)))
            .read_env_overrides(false)
    }
}

#[test]
fn override_env_name_matches_contract() {
    assert_eq!(
        override_env_name("llama3.2", "model"),
        "RMLK_E2E_ASSET_LLAMA3_2_MODEL"
    );
    assert_eq!(
        override_env_name("resnet34", "input"),
        "RMLK_E2E_ASSET_RESNET34_INPUT"
    );
}

#[test]
fn resolves_from_fake_backend_into_cache() {
    let fx = Fixture::new();
    let resolver = fx.resolver();

    let model = resolver
        .resolve(&fx.case_id, &fx.artifact_id, "model")
        .unwrap();
    assert_eq!(fs::read(&model).unwrap(), fx.model_bytes);
    assert!(model.starts_with(&fx.cache));

    let input = resolver
        .resolve(&fx.case_id, &fx.artifact_id, "input")
        .unwrap();
    assert_eq!(fs::read(&input).unwrap(), fx.input_bytes);
}

#[test]
fn cache_hit_skips_backend_after_first_resolve() {
    let fx = Fixture::new();
    let resolver = fx.resolver();
    let first = resolver
        .resolve(&fx.case_id, &fx.artifact_id, "model")
        .unwrap();
    assert_eq!(fs::read(&first).unwrap(), fx.model_bytes);

    // Remove remote payload; cache must still serve.
    fs::remove_file(
        fx.remote
            .join(object_key(&fx.case_id, &fx.artifact_id, "model.onnx")),
    )
    .unwrap();

    let second = resolver
        .resolve(&fx.case_id, &fx.artifact_id, "model")
        .unwrap();
    assert_eq!(second, first);
    assert_eq!(fs::read(&second).unwrap(), fx.model_bytes);
}

#[test]
fn override_wins_over_cache_and_backend() {
    let fx = Fixture::new();
    let override_dir = fx.cache.parent().unwrap().join("override");
    let override_model = override_dir.join("local.onnx");
    write_bytes(&override_model, &fx.model_bytes);

    let resolver = fx
        .resolver()
        .with_override(&fx.case_id, "model", &override_model);

    let path = resolver
        .resolve(&fx.case_id, &fx.artifact_id, "model")
        .unwrap();
    assert_eq!(path, override_model);
}

#[test]
fn corrupt_cache_is_replaced_from_backend() {
    let fx = Fixture::new();
    let key = object_key(&fx.case_id, &fx.artifact_id, "model.onnx");
    let cache_path = fx.cache.join(&key);
    write_bytes(&cache_path, b"not-the-real-bytes");

    let path = fx
        .resolver()
        .resolve(&fx.case_id, &fx.artifact_id, "model")
        .unwrap();
    assert_eq!(path, cache_path);
    assert_eq!(fs::read(&path).unwrap(), fx.model_bytes);
}

#[test]
fn checksum_mismatch_on_override_fails() {
    let fx = Fixture::new();
    let bad = fx.cache.parent().unwrap().join("bad.onnx");
    // Same length as the real model so size check passes and SHA-256 fails.
    let mut wrong = fx.model_bytes.clone();
    wrong[0] ^= 0xff;
    write_bytes(&bad, &wrong);

    let err = fx
        .resolver()
        .with_override(&fx.case_id, "model", &bad)
        .resolve(&fx.case_id, &fx.artifact_id, "model")
        .unwrap_err();
    assert!(matches!(err, ResolveError::ChecksumMismatch { .. }), "{err}");
}

#[test]
fn unknown_role_fails() {
    let fx = Fixture::new();
    let err = fx
        .resolver()
        .resolve(&fx.case_id, &fx.artifact_id, "tokenizer")
        .unwrap_err();
    assert!(matches!(err, ResolveError::UnknownRole { .. }), "{err}");
}

#[test]
fn load_sidecar_parses_pins() {
    let fx = Fixture::new();
    let sidecar: Sidecar = fx
        .resolver()
        .load_sidecar(&fx.case_id, &fx.artifact_id)
        .unwrap();
    assert_eq!(sidecar.onnx_ir_version, 10);
    assert_eq!(sidecar.onnx_opset, 14);
    assert_eq!(sidecar.ort_version, "1.28.0");
    assert!(sidecar.files.contains_key("model"));
}

#[test]
fn from_env_default_cache_root_is_under_repo() {
    let root = common::e2e_assets::default_cache_root();
    assert!(root.ends_with(".cache/rmlk/e2e"));
    assert_eq!(
        override_env_name("resnet34", "sidecar"),
        "RMLK_E2E_ASSET_RESNET34_SIDECAR"
    );
    assert_eq!(
        huggingface_base_url("org/rmlk-e2e", "main"),
        "https://huggingface.co/org/rmlk-e2e/resolve/main"
    );
    // Construct only; no network.
    let _ = HttpBackend::new("https://huggingface.co/org/rmlk-e2e/resolve/main");
    let _ = AssetResolver::<HttpBackend>::from_env();
}

#[test]
fn missing_backend_and_cache_errors() {
    let fx = Fixture::new();
    let resolver: AssetResolver<DirBackend> =
        AssetResolver::new(&fx.cache, None).read_env_overrides(false);
    let err = resolver
        .resolve(&fx.case_id, &fx.artifact_id, "model")
        .unwrap_err();
    assert!(matches!(err, ResolveError::NoBackend { .. }), "{err}");
}
