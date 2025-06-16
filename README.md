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

- The `ConstantOfShape` definition change (in terms of attributes) should be considered in the converter.
- `ReduceMean` axes input should use `usize` type.
- Check how we handle inputs for `ScatterND`.
