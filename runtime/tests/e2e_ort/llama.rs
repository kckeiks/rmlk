//! Llama greedy-decode e2e: short prompt, few tokens, rmlk vs ORT.
//!
//! ORT runs first and is dropped before rmlk loads so only one copy of the
//! ~12 GiB weights sits on the GPU at a time. rmlk then replays the same
//! token feeds (teacher-forced with ORT's next-token ids) and compares
//! last-row logits plus argmax.

use crate::compare::{assert_close_named, max_abs_diff};
use crate::common::e2e_assets::{AssetResolver, HttpBackend};
use crate::runner::{check_sidecar_pins, CaseResult};
use ort::ep;
use ort::memory::Allocator;
use ort::session::Session;
use ort::value::{DynValue, Shape, Tensor};
use rmlk_runtime::{Builder, Value};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

const NUM_LAYERS: usize = 28;
const HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const VOCAB_SIZE: usize = 128256;

const BEGIN_OF_TEXT: &str = "<|begin_of_text|>";
const START_HEADER_ID: &str = "<|start_header_id|>";
const END_HEADER_ID: &str = "<|end_header_id|>";
const EOT_ID: &str = "<|eot_id|>";

/// Stop tokens used by the Llama chat template.
const STOP_TOKEN_IDS: &[usize] = &[128001, 128009];

struct OrtStep {
    next_id: usize,
    last_logits: Vec<f32>,
}

/// Greedy decode for a few tokens; require identical next-token ids and close
/// last-row logits vs ORT CUDA each step.
pub fn run_llama_greedy(
    case_id: &str,
    artifact_id: &str,
    max_new_tokens: usize,
    atol: f32,
    rtol: f32,
) -> Result<CaseResult, String> {
    let resolver = AssetResolver::<HttpBackend>::from_env();
    let sidecar = resolver
        .load_sidecar(case_id, artifact_id)
        .map_err(|e| format!("{case_id}: load sidecar: {e}"))?;
    check_sidecar_pins(&sidecar)?;

    let model_path = resolver
        .resolve(case_id, artifact_id, "model")
        .map_err(|e| format!("{case_id}: resolve model: {e}"))?;
    ensure_external_weights(&resolver, case_id, artifact_id, &model_path)?;

    let tokenizer_path = resolver
        .resolve(case_id, artifact_id, "tokenizer")
        .map_err(|e| format!("{case_id}: resolve tokenizer: {e}"))?;
    let prompt_path = resolver
        .resolve(case_id, artifact_id, "prompt")
        .map_err(|e| format!("{case_id}: resolve prompt: {e}"))?;
    let user_prompt = fs::read_to_string(&prompt_path)
        .map_err(|e| format!("read prompt {}: {e}", prompt_path.display()))?;
    let user_prompt = user_prompt.trim();

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("load tokenizer {}: {e}", tokenizer_path.display()))?;

    let full_prompt = format!(
        "{BEGIN_OF_TEXT}{START_HEADER_ID}system{END_HEADER_ID}\n\
         You are a helpful assistant.{EOT_ID}\n\
         {START_HEADER_ID}user{END_HEADER_ID}\n\
         {user_prompt}{EOT_ID}\n\
         {START_HEADER_ID}assistant{END_HEADER_ID}"
    );
    let enc = tokenizer
        .encode(full_prompt.as_str(), false)
        .map_err(|e| format!("tokenize: {e}"))?;
    let prompt_tokens: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
    if prompt_tokens.is_empty() {
        return Err(format!("{case_id}: empty tokenized prompt"));
    }
    println!(
        "  {case_id}: prompt_tokens={} max_new_tokens={max_new_tokens}",
        prompt_tokens.len()
    );

    let ort_steps = {
        let mut session = Session::builder()
            .map_err(|e| format!("ort Session::builder: {e}"))?
            .with_execution_providers([ep::CUDA::default().build().error_on_failure()])
            .map_err(|e| format!("ort CUDA EP: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("ort commit_from_file: {e}"))?;
        run_ort_greedy(&mut session, &prompt_tokens, max_new_tokens)?
    }; // ORT session dropped here — free GPU weights before rmlk loads.

    let mut engine = Builder::from_onnx_path(&model_path)
        .map_err(|e| format!("rmlk from_onnx_path: {e}"))?
        .build()
        .map_err(|e| format!("rmlk build: {e}"))?;

    let mut max_abs_err = 0.0f32;
    let mut last_token = None;
    let mut past_sequence_len = 0usize;
    let mut input_tokens = prompt_tokens;
    let mut past: Option<HashMap<String, Value>> = None;

    for (step, ort_step) in ort_steps.iter().enumerate() {
        let seq_len = input_tokens.len();
        let feed = build_rmlk_feed(&input_tokens, past_sequence_len, past.take())?;
        past_sequence_len += seq_len;

        let mut out = engine
            .run(feed)
            .map_err(|e| format!("rmlk run step {step}: {e}"))?;

        let logits: Vec<f32> = out
            .remove("logits")
            .ok_or_else(|| format!("{case_id}: rmlk missing logits"))?
            .try_into()
            .map_err(|e| format!("rmlk logits: {e}"))?;
        let rmlk_last = last_logit_row(&logits, seq_len)?;

        assert_close_named(
            &format!("logits_step{step}"),
            rmlk_last,
            &ort_step.last_logits,
            atol,
            rtol,
        )?;
        if let Some(err) = max_abs_diff(rmlk_last, &ort_step.last_logits) {
            max_abs_err = max_abs_err.max(err);
        }

        let rmlk_id = argmax(rmlk_last);
        if rmlk_id != ort_step.next_id {
            return Err(format!(
                "{case_id}: next-token mismatch at step {step}: rmlk={rmlk_id} ort={}",
                ort_step.next_id
            ));
        }
        last_token = Some(rmlk_id);
        println!("  step {step}: next_token={rmlk_id}");

        if STOP_TOKEN_IDS.contains(&rmlk_id) {
            break;
        }

        past = Some(extract_rmlk_presents(&mut out)?);
        // Teacher-force ORT's token so both engines see the same inputs next step.
        input_tokens = vec![ort_step.next_id as i64];
    }

    Ok(CaseResult {
        case_id: case_id.to_string(),
        max_abs_err,
        top1: last_token,
    })
}

fn run_ort_greedy(
    session: &mut Session,
    prompt_tokens: &[i64],
    max_new_tokens: usize,
) -> Result<Vec<OrtStep>, String> {
    let mut steps = Vec::with_capacity(max_new_tokens);
    let mut past_sequence_len = 0usize;
    let mut input_tokens = prompt_tokens.to_vec();
    let mut past: Option<HashMap<String, OrtTensor>> = None;

    for step in 0..max_new_tokens {
        let seq_len = input_tokens.len();
        let feed = build_ort_feed(&input_tokens, past_sequence_len, past.take())?;
        past_sequence_len += seq_len;

        let mut out = run_ort_session(session, feed)?;
        let OrtTensor::F32(logits, _) = out
            .remove("logits")
            .ok_or_else(|| format!("ort missing logits at step {step}"))?
        else {
            return Err(format!("ort logits not f32 at step {step}"));
        };
        let last = last_logit_row(&logits, seq_len)?.to_vec();
        let next_id = argmax(&last);
        steps.push(OrtStep {
            next_id,
            last_logits: last,
        });
        println!("  ort step {step}: next_token={next_id}");

        if STOP_TOKEN_IDS.contains(&next_id) {
            break;
        }

        past = Some(extract_ort_presents(&mut out)?);
        input_tokens = vec![next_id as i64];
    }
    Ok(steps)
}

enum OrtTensor {
    F32(Vec<f32>, Vec<usize>),
    I64(Vec<i64>, Vec<usize>),
}

fn build_rmlk_feed(
    input_tokens: &[i64],
    past_sequence_len: usize,
    past: Option<HashMap<String, Value>>,
) -> Result<HashMap<String, Value>, String> {
    let seq_len = input_tokens.len();
    let input_shape = vec![1usize, seq_len];
    let position_ids: Vec<i64> =
        (past_sequence_len as i64..(past_sequence_len + seq_len) as i64).collect();
    let attention_mask = vec![1i64; past_sequence_len + seq_len];
    let attn_shape = vec![1usize, past_sequence_len + seq_len];

    let mut feed: HashMap<String, Value> = HashMap::new();
    feed.insert(
        "input_ids".into(),
        (input_tokens.to_vec(), input_shape.clone()).into(),
    );
    feed.insert(
        "attention_mask".into(),
        (attention_mask, attn_shape).into(),
    );
    feed.insert(
        "position_ids".into(),
        (position_ids, input_shape).into(),
    );

    if let Some(mut past) = past {
        for layer in 0..NUM_LAYERS {
            let key = past
                .remove(&format!("present.{layer}.key"))
                .ok_or_else(|| format!("rmlk missing present.{layer}.key"))?;
            let val = past
                .remove(&format!("present.{layer}.value"))
                .ok_or_else(|| format!("rmlk missing present.{layer}.value"))?;
            feed.insert(format!("past_key_values.{layer}.key"), key);
            feed.insert(format!("past_key_values.{layer}.value"), val);
        }
    } else {
        let (data, shape) = empty_kv();
        for layer in 0..NUM_LAYERS {
            feed.insert(
                format!("past_key_values.{layer}.key"),
                (data.clone(), shape.clone()).into(),
            );
            feed.insert(
                format!("past_key_values.{layer}.value"),
                (data.clone(), shape.clone()).into(),
            );
        }
    }
    Ok(feed)
}

fn build_ort_feed(
    input_tokens: &[i64],
    past_sequence_len: usize,
    past: Option<HashMap<String, OrtTensor>>,
) -> Result<HashMap<String, OrtTensor>, String> {
    let seq_len = input_tokens.len();
    let input_shape = vec![1usize, seq_len];
    let position_ids: Vec<i64> =
        (past_sequence_len as i64..(past_sequence_len + seq_len) as i64).collect();
    let attention_mask = vec![1i64; past_sequence_len + seq_len];
    let attn_shape = vec![1usize, past_sequence_len + seq_len];

    let mut feed = HashMap::new();
    feed.insert(
        "input_ids".into(),
        OrtTensor::I64(input_tokens.to_vec(), input_shape.clone()),
    );
    feed.insert(
        "attention_mask".into(),
        OrtTensor::I64(attention_mask, attn_shape),
    );
    feed.insert(
        "position_ids".into(),
        OrtTensor::I64(position_ids, input_shape),
    );

    if let Some(mut past) = past {
        for layer in 0..NUM_LAYERS {
            let key = past
                .remove(&format!("present.{layer}.key"))
                .ok_or_else(|| format!("ort missing present.{layer}.key"))?;
            let val = past
                .remove(&format!("present.{layer}.value"))
                .ok_or_else(|| format!("ort missing present.{layer}.value"))?;
            feed.insert(format!("past_key_values.{layer}.key"), key);
            feed.insert(format!("past_key_values.{layer}.value"), val);
        }
    } else {
        let (data, shape) = empty_kv();
        for layer in 0..NUM_LAYERS {
            feed.insert(
                format!("past_key_values.{layer}.key"),
                OrtTensor::F32(data.clone(), shape.clone()),
            );
            feed.insert(
                format!("past_key_values.{layer}.value"),
                OrtTensor::F32(data.clone(), shape.clone()),
            );
        }
    }
    Ok(feed)
}

fn run_ort_session(
    session: &mut Session,
    feed: HashMap<String, OrtTensor>,
) -> Result<HashMap<String, OrtTensor>, String> {
    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(feed.len());
    for (name, tensor) in feed {
        let dyn_val = match tensor {
            OrtTensor::F32(data, shape) => {
                // ort's from_array rejects zero-length dims; empty KV past uses
                // Tensor::new (CreateTensorAsOrtValue) which allows dim 0.
                if shape.iter().any(|&d| d == 0) {
                    let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
                    Tensor::<f32>::new(&Allocator::default(), Shape::from(shape_i64))
                        .map_err(|e| format!("ort empty Tensor `{name}`: {e}"))?
                        .into_dyn()
                } else {
                    let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
                    Tensor::from_array((shape_i64, data))
                        .map_err(|e| format!("ort Tensor `{name}` f32: {e}"))?
                        .into_dyn()
                }
            }
            OrtTensor::I64(data, shape) => {
                let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
                Tensor::from_array((shape_i64, data))
                    .map_err(|e| format!("ort Tensor `{name}` i64: {e}"))?
                    .into_dyn()
            }
        };
        inputs.push((name, dyn_val));
    }

    let outputs = session
        .run(inputs)
        .map_err(|e| format!("ort run: {e}"))?;

    let mut out = HashMap::new();
    for (name, value) in outputs {
        // logits + KV presents are f32 in this model.
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ort extract `{name}`: {e}"))?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        out.insert(name.to_string(), OrtTensor::F32(data.to_vec(), shape));
    }
    Ok(out)
}

fn extract_rmlk_presents(
    out: &mut HashMap<String, Value>,
) -> Result<HashMap<String, Value>, String> {
    let mut past = HashMap::new();
    for layer in 0..NUM_LAYERS {
        let key_name = format!("present.{layer}.key");
        let val_name = format!("present.{layer}.value");
        let key = out
            .remove(&key_name)
            .ok_or_else(|| format!("rmlk missing {key_name}"))?;
        let val = out
            .remove(&val_name)
            .ok_or_else(|| format!("rmlk missing {val_name}"))?;
        past.insert(key_name, key);
        past.insert(val_name, val);
    }
    Ok(past)
}

fn extract_ort_presents(
    out: &mut HashMap<String, OrtTensor>,
) -> Result<HashMap<String, OrtTensor>, String> {
    let mut past = HashMap::new();
    for layer in 0..NUM_LAYERS {
        let key_name = format!("present.{layer}.key");
        let val_name = format!("present.{layer}.value");
        let key = out
            .remove(&key_name)
            .ok_or_else(|| format!("ort missing {key_name}"))?;
        let val = out
            .remove(&val_name)
            .ok_or_else(|| format!("ort missing {val_name}"))?;
        past.insert(key_name, key);
        past.insert(val_name, val);
    }
    Ok(past)
}

fn empty_kv() -> (Vec<f32>, Vec<usize>) {
    (Vec::new(), vec![1, HEADS, 0, HEAD_DIM])
}

fn last_logit_row(logits: &[f32], seq_len: usize) -> Result<&[f32], String> {
    let expected = seq_len * VOCAB_SIZE;
    if logits.len() != expected {
        return Err(format!(
            "logits length {} != seq_len*{VOCAB_SIZE} ({expected})",
            logits.len()
        ));
    }
    Ok(&logits[(seq_len - 1) * VOCAB_SIZE..seq_len * VOCAB_SIZE])
}

fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// External weight file (`model.onnx_data`) must sit beside `model.onnx`.
fn ensure_external_weights(
    resolver: &AssetResolver<HttpBackend>,
    case_id: &str,
    artifact_id: &str,
    model_path: &Path,
) -> Result<(), String> {
    let sibling = model_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("model.onnx_data");
    if sibling.is_file() {
        return Ok(());
    }

    // Try resolving the optional `model_data` role into the model directory.
    let data_path = resolver
        .resolve(case_id, artifact_id, "model_data")
        .map_err(|e| {
            format!(
                "{case_id}: missing external weights beside {} and resolve model_data failed: {e}",
                model_path.display()
            )
        })?;

    if data_path == sibling {
        return Ok(());
    }
    if data_path
        .file_name()
        .is_some_and(|n| n == "model.onnx_data")
        && data_path.parent() == model_path.parent()
    {
        return Ok(());
    }

    // Last resort: symlink beside the model so ONNX external-data lookup works.
    symlink_file(&data_path, &sibling)?;
    Ok(())
}

fn symlink_file(src: &Path, dst: &Path) -> Result<(), String> {
    if dst.exists() {
        return Ok(());
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(src, dst).map_err(|e| {
            format!(
                "symlink {} -> {}: {e}",
                src.display(),
                dst.display()
            )
        })
    }
    #[cfg(not(unix))]
    {
        let _ = (src, dst);
        Err("external weights must sit beside model.onnx (symlink not supported on this OS)".into())
    }
}
