# rmlk

Rusty Machine Learning Kit (rmlk) is an inference runtime written in Rust, with GPU kernels implemented in CUDA C.

### Resnet34 Example

**Input**

![dog](https://github.com/user-attachments/assets/96cc6c82-83cf-4d3d-9bc1-bd754763918b)

![resnet-demo](https://github.com/user-attachments/assets/8f6e11f5-59dc-40dd-bf54-04a1e5fa5eb6)

### LLama 3.2 Example

![llama-demo](https://github.com/user-attachments/assets/78fc83ef-a8a5-4a2a-82b7-8eeef8eb6872)

## Motivation

The goal of this project is to build a high-performance, memory-safe inference runtime with GPU-backed execution.

This project focuses on:

- Leveraging Rust for safety and reliability in production environments
- Executing models on CUDA-enabled GPUs via custom kernels
- Laying the foundation for multi-model and distributed inference

## Vision

The long-term vision for rmlk is to evolve into a distributed inference system where:

- Multiple models can be loaded and executed simultaneously
- Workloads are dynamically balanced across a cluster of GPU devices
- Hardware utilization is maximized through intelligent scheduling

## Current State

The project is functional and under active development.

The runtime is capable of executing ONNX models end-to-end on CUDA-enabled GPUs.

- Supports inference for models compatible with ONNX IR v10 and opset v14.
- Core runtime and CUDA execution pipeline are implemented

### Limitations

- Operator coverage is currently limited
- Execution is currently sequential

### Planned

- Parallel execution of operators
- Multi-model inference support
- Load balancing across devices

The current operations supported are in the table below.


| Operator          | Status   | Notes                  |
|-------------------|----------|------------------------|
| `Expand`          | ✅ Done   |                        |
| `Trilu`           | ✅ Done   |                        |
| `ScatterND`       | ✅ Done   |                        |
| `Greater`         | ✅ Done   |                        |
| `Equal`           | ✅ Done   |                        |
| `Cast`            | ✅ Done   |                        |
| `Pow`             | ✅ Done   |                        |
| `Div`             | ✅ Done   |                        |
| `Sub`             | ✅ Done   |                        |
| `Sqrt`            | ✅ Done   |                        |
| `Neg`             | ✅ Done   |                        |
| `Sin`             | ✅ Done   |                        |
| `Cos`             | ✅ Done   |                        |
| `Softmax`         | ✅ Done   | (with limited support) |
| `MatMul`          | ✅ Done   |                        |
| `Unsqueeze`       | ✅ Done   |                        |
| `Sigmoid`         | ✅ Done   |                        |
| `Relu`            | ✅ Done   |                        |
| `Shape`           | ✅ Done   |                        |
| `ReduceMean`      | ✅ Done   |                        |
| `Gather`          | ✅ Done   |                        |
| `Mul`             | ✅ Done   |                        |
| `Where`           | ✅ Done   |                        |
| `Add`             | ✅ Done   |                        |
| `Slice`           | ✅ Done   |                        |
| `ConstantOfShape` | ✅ Done   |                        |
| `Transpose`       | ✅ Done   |                        |
| `Range`           | ✅ Done   |                        |
| `Concat`          | ✅ Done   |                        |
| `Reshape`         | ✅ Done   |                        |
| `Constant`        | ✅ Done   |                        |

## Testing

Contributor guide (all suites, commands, regression workflow):
[`docs/testing.md`](docs/testing.md).

Compatibility pins (ONNX IR/opset, ORT): [`docs/compatibility.md`](docs/compatibility.md).

Full-model artifact layout: [`docs/e2e-artifacts.md`](docs/e2e-artifacts.md).

### Quick commands

```bash
# Host (CI)
cargo test -p rmlk-schema -p rmlk-graph -p rocky

# GPU op + graph tests
export CUDA_COMPUTE_CAP=89   # if nvidia-smi probe fails
cargo test -p rmlk-runtime --lib
cargo test -p rmlk-runtime --test graphs

# Official ONNX node suite (ignored; needs GPU + oracle Python)
python3 scripts/discover_onnx_node_cases.py
cargo test -p rmlk-runtime --test onnx_node -- --ignored --nocapture

# Full-model e2e vs ORT (ignored; pack artifacts first — see docs/testing.md)
export ORT_CUDA_VERSION=12
CUDA_COMPUTE_CAP=89 cargo test -p rmlk-runtime --test e2e_ort -- --ignored --nocapture
```

### Running Inference

Please see the examples under `runtime`.

### Tested Models

- `resnet34`
- `llama3.2`
  
