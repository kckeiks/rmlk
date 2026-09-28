//! Manual full-model e2e: rmlk vs pinned ORT (CUDA EP).
//!
//! Resolves artifacts (override → cache → Hugging Face / HTTPS), runs the same
//! inputs through ORT and rmlk, and compares float outputs with per-case
//! tolerances. Failures name the output, worst index, and tolerance exceeded.
//!
//! ResNet34 uses a versioned JPEG (`image` role) and ImageNet preprocess in
//! process — see `docs/e2e-artifacts.md` and `scripts/pack_e2e_resnet34.py`.
//! Llama-3.2 uses tokenizer + prompt artifacts and a short greedy decode —
//! see `scripts/pack_e2e_llama32.py`.
//!
//! ```text
//! # pack local artifacts (once):
//! PYTHONPATH=.venv-oracle/lib/python3.12/site-packages \
//!   python3 scripts/pack_e2e_resnet34.py --model ~/Downloads/resnet34.onnx
//! PYTHONPATH=.venv-oracle/lib/python3.12/site-packages \
//!   python3 scripts/pack_e2e_llama32.py \
//!     --model-dir ~/Models/Llama-3.2-3B-Instruct
//!
//! # CUDA 12 hosts (ort 2.0.0-rc.12 ships both; force 12 if auto-detect is wrong):
//! export ORT_CUDA_VERSION=12
//! # If the cuda crate cannot probe the GPU via nvidia-smi, set sm version:
//! CUDA_COMPUTE_CAP=89 cargo test -p rmlk-runtime --test e2e_ort -- --ignored --nocapture
//! ```

#[path = "../common/mod.rs"]
mod common;

mod cases;
mod compare;
mod llama;
mod preprocess;
mod runner;

use cases::Case;

#[test]
#[ignore = "requires CUDA GPU, ORT CUDA EP, and e2e artifacts (Hub or overrides)"]
fn run_full_model_e2e() {
    let cases = cases::all();
    assert!(
        !cases.is_empty(),
        "e2e_ort: case registry is empty"
    );

    let mut failures: Vec<(String, String)> = Vec::new();
    let mut passed = 0usize;

    for case in cases {
        match run_one(case) {
            Ok(result) => {
                passed += 1;
                let top = result
                    .top1
                    .map(|t| format!(" top1={t}"))
                    .unwrap_or_default();
                println!(
                    "ok  {}  max_abs_err={:.6e}{top}",
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
    runner::run_case(case)
}

#[test]
fn cases_are_registered() {
    let cases = cases::all();
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[0].id, "resnet34");
    assert_eq!(cases[0].expect_top1, Some(207));
    assert_eq!(cases[1].id, "llama3.2");
    assert!(matches!(
        cases[1].kind,
        cases::CaseKind::LlamaGreedy { max_new_tokens: 8 }
    ));
}

#[test]
fn huggingface_base_url_shape() {
    let url = common::e2e_assets::huggingface_base_url("org/rmlk-e2e", "main");
    assert_eq!(url, "https://huggingface.co/org/rmlk-e2e/resolve/main");
}
