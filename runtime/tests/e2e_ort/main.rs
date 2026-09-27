//! Manual full-model e2e: rmlk vs pinned ORT (CUDA EP).
//!
//! Resolves artifacts (override → cache → Hugging Face / HTTPS), runs the same
//! inputs through ORT and rmlk, and compares float outputs with per-case
//! tolerances. Failures name the output, worst index, and tolerance exceeded.
//!
//! Not part of the default test run or required PR CI:
//!
//! ```text
//! cargo test -p rmlk-runtime --test e2e_ort -- --ignored --nocapture
//! ```
//!
//! Asset env vars: see `docs/e2e-artifacts.md`. Cases are registered in
//! `cases.rs` (ResNet / Llama land in items 60–61).

#[path = "../common/mod.rs"]
mod common;

mod cases;
mod compare;
mod runner;

use cases::Case;

#[test]
#[ignore = "requires CUDA GPU, ORT CUDA EP, and e2e artifacts (Hub or overrides)"]
fn run_full_model_e2e() {
    let cases = cases::all();
    if cases.is_empty() {
        println!(
            "e2e_ort: no cases registered yet (add ResNet / Llama in items 60–61); nothing to run"
        );
        return;
    }

    let mut failures: Vec<(String, String)> = Vec::new();
    let mut passed = 0usize;

    for case in cases {
        match run_one(case) {
            Ok(result) => {
                passed += 1;
                println!(
                    "ok  {}  max_abs_err={:.6e}",
                    result.case_id, result.max_abs_err
                );
            }
            Err(err) => {
                println!("FAIL  {}: {err}", case.id);
                failures.push((case.id.to_string(), err));
            }
        }
    }

    println!();
    println!("=== full-model e2e summary ===");
    println!("cases:   {}", cases.len());
    println!("passed:  {passed}");
    println!("failed:  {}", failures.len());
    if !failures.is_empty() {
        println!();
        println!("Failures:");
        for (id, err) in &failures {
            println!("  {id}: {err}");
        }
        panic!("{} full-model e2e case(s) failed", failures.len());
    }
}

fn run_one(case: &Case) -> Result<runner::CaseResult, String> {
    let resolver = common::e2e_assets::AssetResolver::<common::e2e_assets::HttpBackend>::from_env();
    let sidecar = resolver
        .load_sidecar(case.id, case.artifact_id)
        .map_err(|e| format!("load sidecar: {e}"))?;
    runner::check_sidecar_pins(&sidecar)?;
    runner::run_case(case)
}

#[test]
fn case_registry_is_wired() {
    // Host-only: registry exists; cases land with items 60–61.
    let _ = cases::all();
}

#[test]
fn huggingface_base_url_shape() {
    let url = common::e2e_assets::huggingface_base_url("org/rmlk-e2e", "main");
    assert_eq!(
        url,
        "https://huggingface.co/org/rmlk-e2e/resolve/main"
    );
}
