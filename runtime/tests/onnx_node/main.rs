//! Official ONNX node-suite conformance (discovered from the pinned `onnx` package).
//!
//! Discovery writes a local cache manifest (not committed). The harness runs every
//! case whose ops are all in rmlk's supported set, compares outputs to ONNX's
//! `.pb` oracles, and prints a summary of passes, failures, and ops that blocked
//! cases from running.
//!
//! ```text
//! python3 scripts/discover_onnx_node_cases.py
//! cargo test -p rmlk-runtime --test onnx_node -- --ignored --nocapture
//! ```

use quick_protobuf::{BytesReader, MessageRead};
use rmlk_runtime::{
    Builder, Value, UNSUPPORTED_CAST_PREFIX, UNSUPPORTED_DATA_TYPE_PREFIX,
};
use rmlk_schema::onnx::TensorProto;
use rmlk_schema::{tensor_from_onnx_tensor, DataType, Tensor};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::Command;

const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-4;
const MANIFEST_VERSION: u64 = 2;

/// ONNX op type strings that rmlk can import today (`Op::from_str` names).
const SUPPORTED_OPS: &[&str] = &[
    "Add",
    "Cast",
    "Concat",
    "Conv",
    "Constant",
    "ConstantOfShape",
    "Cos",
    "Div",
    "Equal",
    "Expand",
    "Flatten",
    "Gather",
    "Gemm",
    "GlobalAveragePool",
    "Greater",
    "MatMul",
    "MaxPool",
    "Mul",
    "Neg",
    "Pow",
    "Range",
    "ReduceMean",
    "Relu",
    "Reshape",
    "ScatterND",
    "Shape",
    "Sigmoid",
    "Sin",
    "Slice",
    "Softmax",
    "Sqrt",
    "Sub",
    "Transpose",
    "Trilu",
    "Unsqueeze",
    "Where",
];

struct CaseEntry {
    name: String,
    ops: Vec<String>,
}

struct Manifest {
    onnx_version: String,
    node_data_root: PathBuf,
    cases: Vec<CaseEntry>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime crate parent")
        .to_path_buf()
}

fn default_cache_dir() -> PathBuf {
    repo_root().join(".cache/rmlk/onnx-node/1.21.0")
}

fn cache_dir() -> PathBuf {
    std::env::var_os("RMLK_ONNX_NODE_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(default_cache_dir)
}

fn run_discover() -> PathBuf {
    let manifest_path = cache_dir().join("manifest.json");
    let script = repo_root().join("scripts/discover_onnx_node_cases.py");
    let status = Command::new("python3")
        .arg(&script)
        .env("RMLK_ONNX_NODE_CACHE", cache_dir())
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", script.display()));
    assert!(
        status.success(),
        "discover_onnx_node_cases.py failed; install pins with \
         `python3 -m pip install -r scripts/requirements-oracle.in`"
    );
    assert!(
        manifest_path.is_file(),
        "manifest missing after discover: {}",
        manifest_path.display()
    );
    manifest_path
}

fn ensure_manifest() -> PathBuf {
    let manifest_path = cache_dir().join("manifest.json");
    if manifest_path.is_file() {
        if let Ok(text) = fs::read_to_string(&manifest_path) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                let version = value["manifest_version"].as_u64().unwrap_or(0);
                if version >= MANIFEST_VERSION {
                    return manifest_path;
                }
            }
        }
    }
    run_discover()
}

fn load_manifest(path: &Path) -> Manifest {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let value: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));

    let onnx_version = value["onnx_version"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let node_data_root = PathBuf::from(
        value["node_data_root"]
            .as_str()
            .unwrap_or_else(|| panic!("{}: missing node_data_root", path.display())),
    );
    let cases = value["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("{}: missing cases", path.display()))
        .iter()
        .map(|c| CaseEntry {
            name: c["name"].as_str().unwrap().to_string(),
            ops: c["ops"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| o.as_str().unwrap().to_string())
                .collect(),
        })
        .collect();

    Manifest {
        onnx_version,
        node_data_root,
        cases,
    }
}

fn supported_op_set() -> BTreeSet<&'static str> {
    SUPPORTED_OPS.iter().copied().collect()
}

fn missing_ops(case_ops: &[String], supported: &BTreeSet<&str>) -> Vec<String> {
    let mut missing: Vec<String> = case_ops
        .iter()
        .filter(|op| !supported.contains(op.as_str()))
        .cloned()
        .collect();
    missing.sort();
    missing.dedup();
    missing
}

/// `Debug` spelling of [`rmlk_schema::Error::NotSupported`] as formatted with `{:?}`.
const SCHEMA_NOT_SUPPORTED_DEBUG: &str = "NotSupported";

/// True when the op was accepted but a data type or cast path is not implemented.
fn is_dtype_or_cast_gap(err: &str) -> bool {
    let e = err.to_lowercase();
    e.contains(UNSUPPORTED_DATA_TYPE_PREFIX)
        || e.contains(UNSUPPORTED_CAST_PREFIX)
        || e.contains(&SCHEMA_NOT_SUPPORTED_DEBUG.to_lowercase())
}

/// Format a payload caught by [`catch_unwind`] (a panic that did not abort).
fn format_catch_unwind_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        format!("caught panic: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("caught panic: {s}")
    } else {
        "caught panic: <non-string payload>".into()
    }
}

fn load_tensor_pb(path: &Path) -> Result<Tensor, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut reader = BytesReader::from_bytes(&bytes);
    let proto = TensorProto::from_reader(&mut reader, &bytes)
        .map_err(|e| format!("parse {}: {e}", path.display()))?;
    tensor_from_onnx_tensor(proto, None)
        .map_err(|e| format!("tensor_from_onnx_tensor {}: {e:?}", path.display()))
}

fn tensor_to_value(tensor: &Tensor) -> Result<Value, String> {
    let dims = tensor.dims.clone();
    match tensor.data_type {
        DataType::Float => {
            let data = if tensor.raw_data.is_some() {
                tensor
                    .to_vec::<f32>()
                    .map_err(|e| format!("f32 raw_data: {e:?}"))?
            } else {
                tensor.float_data.clone()
            };
            Ok((data, dims).into())
        }
        DataType::Int64 => {
            let data = if tensor.raw_data.is_some() {
                tensor
                    .to_vec::<i64>()
                    .map_err(|e| format!("i64 raw_data: {e:?}"))?
            } else {
                tensor.int64_data.clone()
            };
            Ok((data, dims).into())
        }
        DataType::Int32 => {
            let data = if tensor.raw_data.is_some() {
                tensor
                    .to_vec::<i32>()
                    .map_err(|e| format!("i32 raw_data: {e:?}"))?
            } else {
                tensor.int32_data.clone()
            };
            Ok((data, dims).into())
        }
        DataType::Bool => {
            let data: Vec<bool> = if let Some(raw) = tensor.raw_data.as_deref() {
                raw.iter().map(|&b| b != 0).collect()
            } else {
                tensor.int32_data.iter().map(|&v| v != 0).collect()
            };
            Ok((data, dims).into())
        }
        other => Err(format!("{}: {:?}", UNSUPPORTED_DATA_TYPE_PREFIX, other)),
    }
}

fn tensor_name(tensor: &Tensor, fallback: &str) -> String {
    tensor
        .name
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

fn list_numbered(dir: &Path, prefix: &str) -> Result<Vec<PathBuf>, String> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("read_dir {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".pb"))
        })
        .collect();
    paths.sort();
    Ok(paths)
}

fn assert_value_close(output: &str, actual: Value, expected: &Tensor) -> Result<(), String> {
    match expected.data_type {
        DataType::Float => {
            let actual: Vec<f32> = actual
                .try_into()
                .map_err(|e| format!("{output}: actual f32: {e}"))?;
            let expected_data = if expected.raw_data.is_some() {
                expected
                    .to_vec::<f32>()
                    .map_err(|e| format!("{output}: expected f32: {e:?}"))?
            } else {
                expected.float_data.clone()
            };
            if actual.len() != expected_data.len() {
                return Err(format!(
                    "{output}: length mismatch actual={} expected={}",
                    actual.len(),
                    expected_data.len()
                ));
            }
            for (i, (a, e)) in actual.iter().zip(expected_data.iter()).enumerate() {
                let tol = ATOL + RTOL * e.abs();
                if (a - e).abs() > tol {
                    return Err(format!(
                        "{output}: mismatch at {i}: actual={a} expected={e} tol={tol}"
                    ));
                }
            }
            Ok(())
        }
        DataType::Int64 => {
            let actual: Vec<i64> = actual
                .try_into()
                .map_err(|e| format!("{output}: actual i64: {e}"))?;
            let expected_data = if expected.raw_data.is_some() {
                expected
                    .to_vec::<i64>()
                    .map_err(|e| format!("{output}: expected i64: {e:?}"))?
            } else {
                expected.int64_data.clone()
            };
            if actual != expected_data {
                return Err(format!("{output}: i64 mismatch"));
            }
            Ok(())
        }
        DataType::Int32 => {
            let actual: Vec<i32> = actual
                .try_into()
                .map_err(|e| format!("{output}: actual i32: {e}"))?;
            let expected_data = if expected.raw_data.is_some() {
                expected
                    .to_vec::<i32>()
                    .map_err(|e| format!("{output}: expected i32: {e:?}"))?
            } else {
                expected.int32_data.clone()
            };
            if actual != expected_data {
                return Err(format!("{output}: i32 mismatch"));
            }
            Ok(())
        }
        DataType::Bool => {
            let actual: Vec<bool> = actual
                .try_into()
                .map_err(|e| format!("{output}: actual bool: {e}"))?;
            let expected_data: Vec<bool> = if expected.raw_data.is_some() {
                expected
                    .raw_data
                    .as_deref()
                    .unwrap()
                    .iter()
                    .map(|&b| b != 0)
                    .collect()
            } else {
                expected.int32_data.iter().map(|&v| v != 0).collect()
            };
            if actual != expected_data {
                return Err(format!("{output}: bool mismatch"));
            }
            Ok(())
        }
        other => Err(format!("{output}: unsupported expected dtype {other:?}")),
    }
}

fn run_case(case_dir: &Path) -> Result<(), String> {
    let model_path = case_dir.join("model.onnx");
    let model_bytes =
        fs::read(&model_path).map_err(|e| format!("read {}: {e}", model_path.display()))?;

    let builder = Builder::from_onnx_bytes(&model_bytes)
        .map_err(|e| format!("from_onnx_bytes failed: {e}"))?;
    let mut instance = builder
        .build()
        .map_err(|e| format!("build failed: {e}"))?;

    let mut data_sets: Vec<PathBuf> = fs::read_dir(case_dir)
        .map_err(|e| format!("read_dir {}: {e}", case_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("test_data_set_"))
        })
        .collect();
    data_sets.sort();
    if data_sets.is_empty() {
        return Err("no test_data_set_* directories".into());
    }

    for data_set in data_sets {
        let mut feeds = HashMap::new();
        for (idx, path) in list_numbered(&data_set, "input_")?.into_iter().enumerate() {
            let tensor = load_tensor_pb(&path)?;
            let name = tensor_name(&tensor, &format!("input_{idx}"));
            feeds.insert(name, tensor_to_value(&tensor)?);
        }

        let mut outputs = instance
            .run(feeds)
            .map_err(|e| format!("run failed on {}: {e}", data_set.display()))?;

        for (idx, path) in list_numbered(&data_set, "output_")?.into_iter().enumerate() {
            let expected = load_tensor_pb(&path)?;
            let name = tensor_name(&expected, &format!("output_{idx}"));
            let actual = outputs
                .remove(&name)
                .ok_or_else(|| format!("missing output `{name}`"))?;
            assert_value_close(&name, actual, &expected)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA GPU and pinned onnx; run with --ignored"]
fn run_discovered_node_cases() {
    let manifest_path = ensure_manifest();
    let manifest = load_manifest(&manifest_path);
    let supported_ops = supported_op_set();

    assert!(
        manifest.node_data_root.is_dir(),
        "node_data_root missing: {} (re-run scripts/discover_onnx_node_cases.py)",
        manifest.node_data_root.display()
    );

    let mut runnable = Vec::new();
    let mut skipped_unsupported_ops = 0usize;
    let mut skipped_known_broken = 0usize;
    let mut op_blocker_counts: BTreeMap<String, usize> = BTreeMap::new();

    for case in &manifest.cases {
        // Packed TENSOR attributes (e.g. ConstantOfShape value) abort quick-protobuf
        // on unaligned packed fixed reads; skip until item 45 hardens the parser.
        if case.ops.iter().any(|op| op == "ConstantOfShape") {
            skipped_known_broken += 1;
            continue;
        }
        let missing = missing_ops(&case.ops, &supported_ops);
        if missing.is_empty() {
            runnable.push(case);
        } else {
            skipped_unsupported_ops += 1;
            for op in missing {
                *op_blocker_counts.entry(op).or_default() += 1;
            }
        }
    }

    let mut passed = 0usize;
    let mut dtype_cast_gaps: Vec<(String, String)> = Vec::new();
    let mut failures: Vec<(String, String)> = Vec::new();

    for case in &runnable {
        let case_dir = manifest.node_data_root.join(&case.name);
        let result = catch_unwind(AssertUnwindSafe(|| run_case(&case_dir)));
        match result {
            Ok(Ok(())) => {
                passed += 1;
                println!("ok        {}", case.name);
            }
            Ok(Err(err)) if is_dtype_or_cast_gap(&err) => {
                println!("dtype_gap {}: {err}", case.name);
                dtype_cast_gaps.push((case.name.clone(), err));
            }
            Ok(Err(err)) => {
                println!("FAIL      {}: {err}", case.name);
                failures.push((case.name.clone(), err));
            }
            Err(payload) => {
                let err = format_catch_unwind_payload(payload);
                if is_dtype_or_cast_gap(&err) {
                    println!("dtype_gap {}: {err}", case.name);
                    dtype_cast_gaps.push((case.name.clone(), err));
                } else {
                    println!("FAIL      {}: {err}", case.name);
                    failures.push((case.name.clone(), err));
                }
            }
        }
    }

    println!();
    println!("=== ONNX node suite summary (onnx=={}) ===", manifest.onnx_version);
    println!("discovered:              {}", manifest.cases.len());
    println!("runnable (ops known):    {}", runnable.len());
    println!("skipped (unknown op):    {skipped_unsupported_ops}");
    println!("skipped (known broken):  {skipped_known_broken}");
    println!("passed:                  {passed}");
    println!("dtype/cast gap:          {}", dtype_cast_gaps.len());
    println!("failed:                  {}", failures.len());
    if !op_blocker_counts.is_empty() {
        println!();
        println!("Top unknown ops (cases skipped; op not in rmlk):");
        let mut ranked: Vec<_> = op_blocker_counts.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (op, count) in ranked.into_iter().take(40) {
            println!("  {count:>5}  {op}");
        }
    }
    if !dtype_cast_gaps.is_empty() {
        println!();
        println!("Dtype/cast gaps (op is known; dtype or cast path not implemented):");
        for (name, err) in dtype_cast_gaps.iter().take(60) {
            println!("  {name}: {err}");
        }
        if dtype_cast_gaps.len() > 60 {
            println!("  ... and {} more", dtype_cast_gaps.len() - 60);
        }
    }
    if !failures.is_empty() {
        println!();
        println!("Failures:");
        for (name, err) in &failures {
            println!("  {name}: {err}");
        }
        panic!(
            "{} runnable node case(s) failed; see summary above",
            failures.len()
        );
    }
}
