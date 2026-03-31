# rmlk

Rusty Machine Learning Kit (rmlk) is an inference runtime written in Rust, with GPU kernels implemented in CUDA C.

### Resnet34 Example


### LLama 3.2 Example


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

The runtime only supports `.rmlk` files. You must convert your `.onnx` model file to a `.rmlk` file. There is a command-line tool, `rocky`, that performs this conversion.

```bash
$ rocky transform <ONNX_FILE> [OUTPUT]
```

### Running Inference

Please see the examples under `runtime`.

### Tested Models

- `resnet34`
- `llama3.2`

## Notes

Based on ONNX version: 1.13.1