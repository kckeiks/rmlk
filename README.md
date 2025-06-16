# rmlk

Rusty Machine Learning Kit is a machine learning library.

Based on ONNX version: 1.13.1

## Todo Checklist

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

## Notes
- During model loading, the inputs should be loaded in the order expected by each backend component. This order should be formalized in a future internal specification.
- Let's make a better interface for managing tensor shapes and strides that doesn't require a mutable execution state object.
- When should we load cuda kernel functions? Sometimes we need attribute information to decide which kernel type to load.