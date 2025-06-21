use crate::attributes;
use crate::attributes::{cast, reduce_mean, softmax, transpose, trilu};
use crate::core::error::InternalError;
use crate::core::{Context, Tensor};
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use base64::Engine;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{env, fs::File, io::Write};

const ROOT_PATH_DEBUGGER: &str = "/home/blackcat/Documents";

// Todo: handle scalars.
// Users may want a completely empty tensor but this doesnt do that.
pub fn init_tensor_device_data<T>(
    stream: &Arc<CudaStream>,
    mut tensor: Tensor<CudaData>,
) -> Result<()>
where
    T: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let size = tensor.shape().iter().copied().product::<usize>();

    // Allocate device data for the tensor if we haven't done it yet
    // or if the existing allocated data has a different size.
    let dev_data_ref = tensor.dev_data_ptr_mut();
    let need_to_alloc_dev_data = dev_data_ref.is_none()
        || dev_data_ref
            .as_ref()
            .map(|data| data.data::<T>().len() != size)
            .unwrap_or(true);

    // We need to remove this immutable reference so we can mutate `y`.
    drop(dev_data_ref);

    if need_to_alloc_dev_data {
        let dev_data = stream
            .alloc_zeros::<T>(size)
            .map_err(rmlk_cuda::Error::from)?;
        tensor.set_dev_data(CudaData::new(dev_data));
    };

    Ok(())
}

pub fn init_tensor_device_data_with_empty_slice<T>(
    stream: &Arc<CudaStream>,
    mut tensor: Tensor<CudaData>,
) -> Result<()>
where
    T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
{
    let dev_data = stream.alloc_zeros::<T>(0).map_err(rmlk_cuda::Error::from)?;
    tensor.set_dev_data(CudaData::new(dev_data));

    Ok(())
}

pub fn copy_tensor_dev_data<T>(
    stream: &Arc<CudaStream>,
    src: &Tensor<CudaData>,
    dst: &mut Tensor<CudaData>,
) -> Result<()>
where
    T: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let src_dev_ptr = src.try_dev_data_ptr()?;
    let src = src_dev_ptr.data::<T>();

    let mut dev_data = unsafe {
        stream
            .alloc::<T>(src.len())
            .map_err(rmlk_cuda::Error::from)?
    };
    stream
        .memcpy_dtod(src.as_ref(), &mut dev_data)
        .map_err(|e| InternalError::Device { error: e.into() })?;
    dst.set_dev_data(CudaData::new(dev_data));

    Ok(())
}

fn numpy_dtype(dt: DataType) -> &'static str {
    match dt {
        DataType::Float => "<f4",
        DataType::Double => "<f8",
        DataType::Int8 => "|i1",
        DataType::Uint8 => "|u1",
        DataType::Int16 => "<i2",
        DataType::Int32 => "<i4",
        DataType::Int64 => "<i8",
        DataType::Bool => "|b1",
        _ => panic!("unsupported data type"),
    }
}

/// Copy any `[T]` into a `Vec<u8>` in *native* byte order (little-endian on
/// x86/ARM).  Requires only that `T` is `Copy`.
fn slice_to_bytes<T>(data: &[T]) -> Vec<u8> {
    let byte_len = data.len() * std::mem::size_of::<T>();
    let ptr = data.as_ptr() as *const u8;
    // SAFETY: `data` is valid for `byte_len` bytes and remains so for the
    // lifetime of the slice we immediately copy with `.to_vec()`.
    unsafe { std::slice::from_raw_parts(ptr, byte_len) }.to_vec()
}

fn value_from_tensor<T>(stream: Arc<CudaStream>, tensor: &Tensor<CudaData>) -> Value
where
    T: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let tensor_ptr = tensor.try_dev_data_ptr().unwrap();
    let tensor_view = tensor_ptr.data::<T>();

    let data = stream.memcpy_dtov(tensor_view.as_ref()).unwrap();

    json!({
        "shape":  tensor.shape(),
        "stride": tensor.stride(),
        "dtype":  numpy_dtype(tensor.dtype()),
        "data":  base64::engine::general_purpose::STANDARD.encode(slice_to_bytes(&data)),
    })
}

pub fn write_results_ternary_two_optional<A, B, C, D, OUT>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
    attributes: HashMap<String, String>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    B: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    C: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    D: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    OUT: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let op_name = ctx.get_node().unwrap().value().name().unwrap();

    let mut dump = Map::new();
    dump.insert("op".into(), json!(op_name));

    let a = ctx.get_input(0)?;
    let b = ctx.get_input(1)?;
    let c = ctx.get_input(2)?;

    let mut inputs = vec![
        value_from_tensor::<A>(stream.clone(), &a),
        value_from_tensor::<B>(stream.clone(), &b),
        value_from_tensor::<C>(stream.clone(), &c),
    ];

    if let Ok(input) = ctx.get_input(3) {
        inputs.push(value_from_tensor::<D>(stream.clone(), &input))
    }

    if let Ok(input) = ctx.get_input(4) {
        inputs.push(value_from_tensor::<D>(stream.clone(), &input))
    }

    dump.insert("inputs".into(), Value::Array(inputs));

    let output = ctx.get_output(0)?;

    let outputs = vec![value_from_tensor::<OUT>(stream, &output)];
    dump.insert("outputs".into(), Value::Array(outputs));

    let attrs_json: Map<String, Value> = attributes
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect();
    dump.insert("attributes".into(), Value::Object(attrs_json));

    let ran_str = random_string(op_name.len());
    // let mut file = File::create(trace_path(ROOT_PATH_DEBUGGER, &path, op_name, &ran_str))?;
    // file.write_all(serde_json::to_string_pretty(&dump)?.as_bytes())?;
    Ok(())
}

pub fn write_results_ternary<A, B, C, OUT>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
    attributes: HashMap<String, String>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    B: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    C: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    OUT: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let op_name = ctx.get_node().unwrap().value().name().unwrap();

    let a = ctx.get_input(0)?;
    let b = ctx.get_input(1)?;
    let c = ctx.get_input(2)?;
    let output = ctx.get_output(0)?;

    let mut dump = Map::new();
    dump.insert("op".into(), json!(op_name));

    let inputs = vec![
        value_from_tensor::<A>(stream.clone(), &a),
        value_from_tensor::<B>(stream.clone(), &b),
        value_from_tensor::<C>(stream.clone(), &c),
    ];
    dump.insert("inputs".into(), Value::Array(inputs));

    let outputs = vec![value_from_tensor::<OUT>(stream, &output)];
    dump.insert("outputs".into(), Value::Array(outputs));

    let attrs_json: Map<String, Value> = attributes
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect();
    dump.insert("attributes".into(), Value::Object(attrs_json));

    let ran_str = random_string(op_name.len());
    // let mut file = File::create(trace_path(ROOT_PATH_DEBUGGER, &path, op_name, &ran_str))?;
    // file.write_all(serde_json::to_string_pretty(&dump)?.as_bytes())?;
    Ok(())
}

pub fn write_results_binary<A, B, OUT>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
    attributes: HashMap<String, String>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    B: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    OUT: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let op_name = ctx.get_node().unwrap().value().name().unwrap();

    let a = ctx.get_input(0)?;
    let b = ctx.get_input(1)?;
    let output = ctx.get_output(0)?;

    let mut dump = Map::new();
    dump.insert("op".into(), json!(op_name));

    let inputs = vec![
        value_from_tensor::<A>(stream.clone(), &a),
        value_from_tensor::<B>(stream.clone(), &b),
    ];
    dump.insert("inputs".into(), Value::Array(inputs));

    let outputs = vec![value_from_tensor::<OUT>(stream, &output)];
    dump.insert("outputs".into(), Value::Array(outputs));

    let attrs_json: Map<String, Value> = attributes
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect();
    dump.insert("attributes".into(), Value::Object(attrs_json));

    let ran_str = random_string(op_name.len());
    // let mut file = File::create(trace_path(ROOT_PATH_DEBUGGER, &path, op_name, &ran_str))?;
    // file.write_all(serde_json::to_string_pretty(&dump)?.as_bytes())?;
    Ok(())
}

pub fn write_results_binary_with_optional_second<A, B, OUT>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
    attributes: HashMap<String, String>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    B: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    OUT: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let parent_path = env::var("CARGO_MANIFEST_DIR").unwrap();
    let op_name = ctx.get_node().unwrap().value().name().unwrap();

    let mut dump = Map::new();
    dump.insert("op".into(), json!(op_name));

    let a = ctx.get_input(0)?;
    let mut inputs = vec![value_from_tensor::<A>(stream.clone(), &a)];

    if let Ok(b) = ctx.get_input(1) {
        inputs.push(value_from_tensor::<B>(stream.clone(), &b));
    }

    dump.insert("inputs".into(), Value::Array(inputs));

    let output = ctx.get_output(0)?;
    let outputs = vec![value_from_tensor::<OUT>(stream, &output)];
    dump.insert("outputs".into(), Value::Array(outputs));

    let attrs_json: Map<String, Value> = attributes
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect();
    dump.insert("attributes".into(), Value::Object(attrs_json));

    let ran_str = random_string(op_name.len());
    // let mut file = File::create(trace_path(ROOT_PATH_DEBUGGER, &path, op_name, &ran_str))?;
    // file.write_all(serde_json::to_string_pretty(&dump)?.as_bytes())?;
    Ok(())
}

pub fn write_results_unary<A, OUT>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
    attributes: HashMap<String, String>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    OUT: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let parent_path = env::var("CARGO_MANIFEST_DIR").unwrap();
    let op_name = ctx.get_node().unwrap().value().name().unwrap();

    let a = ctx.get_input(0)?;
    let output = ctx.get_output(0)?;

    let mut dump = Map::new();
    dump.insert("op".into(), json!(op_name));

    let inputs = vec![value_from_tensor::<A>(stream.clone(), &a)];
    dump.insert("inputs".into(), Value::Array(inputs));

    let outputs = vec![value_from_tensor::<OUT>(stream, &output)];
    dump.insert("outputs".into(), Value::Array(outputs));

    let attrs_json: Map<String, Value> = attributes
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect();
    dump.insert("attributes".into(), Value::Object(attrs_json));

    let ran_str = random_string(op_name.len());
    // let mut file = File::create(trace_path(ROOT_PATH_DEBUGGER, &path, op_name, &ran_str))?;
    // file.write_all(serde_json::to_string_pretty(&dump)?.as_bytes())?;
    Ok(())
}

pub fn write_results_trilu<T>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    T: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    write_results_binary_with_optional_second::<T, i64, T>(path, stream, ctx, Default::default())
}

pub fn write_results_cast<A, OUT>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    OUT: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let mut attrs = HashMap::new();

    if let Some(out_dtype) = ctx
        .get_attributes()
        .map(|attrs| cast::get_value(&attrs))
        .transpose()
        .map(Option::flatten)
        .unwrap()
    {
        attrs.insert("to".to_string(), numpy_dtype(out_dtype).to_string());
    }

    write_results_unary::<A, OUT>(path, stream, ctx, attrs)
}

pub fn write_results_softmax<A, OUT>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    OUT: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let mut attrs = HashMap::new();

    if let Some(axis) = ctx.get_attributes().map(|attrs| softmax::get_axis(&attrs)) {
        attrs.insert("axis".to_string(), format!("{axis}").to_string());
    }

    write_results_unary::<A, OUT>(path, stream, ctx, attrs)
}

pub fn write_results_concat<T>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    T: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let parent_path = env::var("CARGO_MANIFEST_DIR").unwrap();
    let op_name = ctx.get_node().unwrap().value().name().unwrap();

    let mut dump = Map::new();
    dump.insert("op".into(), json!(op_name));

    let mut inputs = vec![];
    for i in 0..usize::MAX {
        match ctx.get_input(i) {
            Ok(input) => {
                inputs.push(value_from_tensor::<T>(stream.clone(), &input));
            }
            Err(_) => break,
        }
    }
    dump.insert("inputs".into(), Value::Array(inputs));

    let output = ctx.get_output(0)?;

    let outputs = vec![value_from_tensor::<T>(stream, &output)];
    dump.insert("outputs".into(), Value::Array(outputs));

    let attrs = ctx.get_attributes().unwrap();
    let axis = attributes::concat::get_axis(&attrs).unwrap();

    let mut attrs_json: Map<String, Value> = Map::new();
    attrs_json.insert("axis".to_string(), Value::String(format!("{:?}", axis)));
    dump.insert("attributes".into(), Value::Object(attrs_json));

    let ran_str = random_string(op_name.len());
    // let mut file = File::create(trace_path(ROOT_PATH_DEBUGGER, &path, op_name, &ran_str))?;
    // file.write_all(serde_json::to_string_pretty(&dump)?.as_bytes())?;
    Ok(())
}

pub fn write_results_gather<A, B>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    B: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let mut attrs = HashMap::new();
    if let Some(axis) = ctx
        .get_attributes()
        .map(|attrs| attributes::gather::get_axis(&attrs))
    {
        attrs.insert("axis".to_string(), format!("{axis}").to_string());
    }
    write_results_binary::<A, B, A>(path, stream, ctx, attrs)
}

pub fn write_results_reduce_mean<A, B>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    B: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let mut attrs = HashMap::new();

    if let Some(keep_dims) = ctx
        .get_attributes()
        .map(|attrs| reduce_mean::get_keep_dims(&attrs))
    {
        attrs.insert("keepdims".to_string(), format!("{keep_dims}").to_string());
    }

    if let Some(noop_with_empty_axes) = ctx
        .get_attributes()
        .as_ref()
        .map(|attrs| reduce_mean::get_noop_with_empty_axes(&attrs))
    {
        attrs.insert(
            "noop_with_empty_axes".to_string(),
            format!("{noop_with_empty_axes}").to_string(),
        );
    }

    if let Some(Some(axes)) = ctx
        .get_attributes()
        .as_ref()
        .map(|attrs| reduce_mean::get_axes(&attrs))
    {
        attrs.insert("axes".to_string(), format!("{axes:?}").to_string());
    }

    write_results_binary_with_optional_second::<A, B, A>(path, stream, ctx, attrs)
}

pub fn write_results_reshape<A>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let mut attrs = HashMap::new();

    match ctx.get_attributes() {
        Some(raw_attrs) => {
            let allow_zero = attributes::reshape::get_allow_zero(&raw_attrs);
            attrs.insert("allowzero".to_string(), format!("{allow_zero}").to_string());
        }
        None => {}
    };

    write_results_binary_with_optional_second::<A, i64, A>(path, stream, ctx, attrs)
}

pub fn write_results_scatter_nd<A>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let raw_attrs = ctx.get_attributes();
    let reduction = match raw_attrs
        .as_ref()
        .and_then(|attrs| attrs.get("reduction"))
        .and_then(|attr| attr.string())
    {
        None => "none".to_string(),
        Some(b) => String::from_utf8_lossy(b).to_string(),
    };
    let mut attrs = HashMap::new();
    attrs.insert("reduction".to_string(), reduction);
    write_results_ternary::<A, i64, A, A>(path, stream, ctx, attrs)
}

pub fn write_results_shape<A>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let mut attrs = HashMap::new();

    if let Some(raw_start) = ctx
        .get_attributes()
        .as_ref()
        .map(|attrs| attributes::shape::get_start(attrs.as_ref()))
    {
        attrs.insert("start".to_string(), format!("{raw_start}").to_string());
    }

    if let Some(raw_end) = ctx
        .get_attributes()
        .and_then(|attrs| attributes::shape::get_end(&attrs))
    {
        attrs.insert("end".to_string(), format!("{raw_end}").to_string());
    }

    write_results_unary::<A, i64>(path, stream, ctx, attrs)
}

pub fn write_results_slice<T, Tind>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    T: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    Tind: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    write_results_ternary_two_optional::<T, Tind, Tind, Tind, T>(
        path,
        stream,
        ctx,
        Default::default(),
    )
}

pub fn write_results_transpose<A>(
    path: &'static str,
    stream: Arc<CudaStream>,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    A: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let mut attrs = HashMap::new();

    if let Some(perm) = ctx
        .get_attributes()
        .as_ref()
        .and_then(|attrs| transpose::get_perm(&attrs))
    {
        attrs.insert("perm".to_string(), format!("{perm:?}").to_string());
    }

    write_results_unary::<A, A>(path, stream, ctx, attrs)
}

fn random_string(len: usize) -> String {
    const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
          abcdefghijklmnopqrstuvwxyz\
          0123456789";

    // simple 64-bit state seeded from the current time
    let mut x = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut out = String::with_capacity(len);

    for _ in 0..len {
        // xorshift64* PRNG (10 ns per iteration on modern CPUs)
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x = x.wrapping_mul(0x2545F4914F6CDD1D);

        out.push(ALPHA[(x as usize) % ALPHA.len()] as char);
    }
    out
}

fn sanitize_op_name(name: &str) -> String {
    name.trim_start_matches(|c| c == '/' || c == '\\') // no leading sep
        .chars()
        .map(|c| match c {
            '/' | '\\' => '_', // flatten dirs
            _ => c,
        })
        .collect()
}

fn trace_path(parent: &str, subdir: &str, op_name: &str, rand_str: &str) -> PathBuf {
    let clean = sanitize_op_name(op_name);

    let p = Path::new(parent)
        .join(subdir)
        .join(format!("{clean}-{rand_str}.json"));

    p
}
