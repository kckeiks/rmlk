//! Shared rmlk vs ORT runner for full-model e2e cases.

use crate::cases::Case;
use crate::compare::assert_close_named;
use crate::common::e2e_assets::{AssetResolver, FileEntry, HttpBackend, Sidecar};
use ort::ep;
use ort::session::Session;
use ort::value::Tensor;
use rmlk_runtime::{Builder, Value};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct CaseResult {
    pub case_id: String,
    pub max_abs_err: f32,
}

pub fn run_case(case: &Case) -> Result<CaseResult, String> {
    let resolver = AssetResolver::<HttpBackend>::from_env();
    let sidecar = resolver
        .load_sidecar(case.id, case.artifact_id)
        .map_err(|e| format!("{}: load sidecar: {e}", case.id))?;

    let model_path = resolver
        .resolve(case.id, case.artifact_id, "model")
        .map_err(|e| format!("{}: resolve model: {e}", case.id))?;

    let mut feeds_rmlk: HashMap<String, Value> = HashMap::new();
    let mut feeds_ort: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    for binding in case.inputs {
        let path = resolver
            .resolve(case.id, case.artifact_id, binding.role)
            .map_err(|e| format!("{}: resolve {}: {e}", case.id, binding.role))?;
        let entry = sidecar
            .files
            .get(binding.role)
            .ok_or_else(|| format!("{}: sidecar missing role {}", case.id, binding.role))?;
        let (data, shape) = load_f32_tensor(&path, entry)?;
        feeds_ort.insert(binding.name.to_string(), (data.clone(), shape.clone()));
        feeds_rmlk.insert(binding.name.to_string(), (data, shape).into());
    }

    let ort_outputs = run_ort_cuda(&model_path, &feeds_ort)?;
    let mut rmlk_outputs = run_rmlk(&model_path, feeds_rmlk)?;

    let mut max_abs_err = 0.0f32;
    for out in case.outputs {
        let expected = ort_outputs
            .get(out.name)
            .ok_or_else(|| format!("{}: ORT missing output `{}`", case.id, out.name))?;
        let actual_val = rmlk_outputs
            .remove(out.name)
            .ok_or_else(|| format!("{}: rmlk missing output `{}`", case.id, out.name))?;
        let actual: Vec<f32> = actual_val
            .try_into()
            .map_err(|e| format!("{}: rmlk output `{}` as f32: {e}", case.id, out.name))?;
        if let Some(err) = max_abs_diff(&actual, expected) {
            max_abs_err = max_abs_err.max(err);
        }
        assert_close_named(out.name, &actual, expected, case.atol, case.rtol)?;
    }

    Ok(CaseResult {
        case_id: case.id.to_string(),
        max_abs_err,
    })
}

fn run_rmlk(
    model_path: &Path,
    feeds: HashMap<String, Value>,
) -> Result<HashMap<String, Value>, String> {
    let builder = Builder::from_onnx_path(model_path).map_err(|e| format!("rmlk from_onnx_path: {e}"))?;
    let mut instance = builder.build().map_err(|e| format!("rmlk build: {e}"))?;
    instance
        .run(feeds)
        .map_err(|e| format!("rmlk run: {e}"))
}

fn run_ort_cuda(
    model_path: &Path,
    feeds: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut session = Session::builder()
        .map_err(|e| format!("ort Session::builder: {e}"))?
        .with_execution_providers([ep::CUDA::default().build().error_on_failure()])
        .map_err(|e| format!("ort CUDA EP: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("ort commit_from_file: {e}"))?;

    let mut ort_inputs: Vec<(String, Tensor<f32>)> = Vec::with_capacity(feeds.len());
    for (name, (data, shape)) in feeds {
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let tensor = Tensor::from_array((shape_i64, data.clone()))
            .map_err(|e| format!("ort Tensor `{name}`: {e}"))?;
        ort_inputs.push((name.clone(), tensor));
    }

    let outputs = session
        .run(ort_inputs)
        .map_err(|e| format!("ort run: {e}"))?;

    let mut out = HashMap::new();
    for (name, value) in outputs {
        let (_shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ort extract `{name}`: {e}"))?;
        out.insert(name.to_string(), data.to_vec());
    }
    Ok(out)
}

fn load_f32_tensor(path: &Path, entry: &FileEntry) -> Result<(Vec<f32>, Vec<usize>), String> {
    let dtype = entry.dtype.as_deref().unwrap_or("f32");
    if dtype != "f32" && dtype != "float" && dtype != "float32" {
        return Err(format!(
            "{}: unsupported dtype `{dtype}` (harness currently loads f32 only)",
            path.display()
        ));
    }
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() as u64 != entry.bytes {
        return Err(format!(
            "{}: size {} != sidecar {}",
            path.display(),
            bytes.len(),
            entry.bytes
        ));
    }
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: not a multiple of 4 bytes", path.display()));
    }
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let shape = entry
        .shape
        .as_ref()
        .map(|s| s.iter().map(|&d| d as usize).collect::<Vec<_>>())
        .ok_or_else(|| format!("{}: sidecar entry missing shape", path.display()))?;
    let expected_len: usize = shape.iter().product();
    if data.len() != expected_len {
        return Err(format!(
            "{}: element count {} != shape product {expected_len} ({shape:?})",
            path.display(),
            data.len()
        ));
    }
    Ok((data, shape))
}

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> Option<f32> {
    if actual.len() != expected.len() {
        return None;
    }
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}


/// Validate sidecar pin fields against the compatibility profile (soft check).
pub fn check_sidecar_pins(sidecar: &Sidecar) -> Result<(), String> {
    const WANT_IR: u64 = 10;
    const WANT_OPSET: u64 = 14;
    const WANT_ORT: &str = "1.28.0";
    if sidecar.onnx_ir_version != WANT_IR {
        return Err(format!(
            "sidecar onnx_ir_version {} != {WANT_IR} (see docs/compatibility.md)",
            sidecar.onnx_ir_version
        ));
    }
    if sidecar.onnx_opset != WANT_OPSET {
        return Err(format!(
            "sidecar onnx_opset {} != {WANT_OPSET} (see docs/compatibility.md)",
            sidecar.onnx_opset
        ));
    }
    if sidecar.ort_version != WANT_ORT {
        return Err(format!(
            "sidecar ort_version {} != {WANT_ORT} (see docs/compatibility.md)",
            sidecar.ort_version
        ));
    }
    Ok(())
}
