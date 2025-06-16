# rmlk

Rusty Machine Learning Kit is a machine learning library.

Based on ONNX version: 1.13.1

## Todo Checklist

| Operator          | Status                        | Notes                            |
|-------------------|-------------------------------|----------------------------------|
| `Expand`          | ✅ Done                        | Needs backend and more tes       |
| `Trilu`           | ✅ Done                        | Needs backend and more tests     |
| `ScatterND`       | ✅ Done                        | Needs backend and more tests     |
| `Greater`         | ✅ Done                        | Needs kernel, backend, and tests |
| `Equal`           | ✅ Done                        | Needs kernel, backend, and tests |
| `Cast`            | ✅ Done                        | Needs backend and tests          |
| `Pow`             | ✅ Done                | Needs kernel, backend, and tests |
| `Div`             | ✅ Done                    | Needs kernel, backend, and tests |
| `Sub`             | 🛠️ In Progress               | Needs kernel, backend, and tests |
| `Sqrt`            | ✅ Done                | Needs backend and tests          |
| `Neg`             | 🛠️ In Progress               | Needs kernel, backend, and tests |
| `Sin`             | 🛠️ In Progress               | Needs kernel, backend, and tests |
| `Cos`             | 🛠️ In Progress               | Needs kernel, backend, and tests |
| `Softmax`         | ✅ Done (with limited support) |                                  |
| `MatMul`          | ✅ Done                        |                                  |
| `Unsqueeze`       | ✅ Done                        |                                  |
| `Sigmoid`         | ✅ Done                        |                                  |
| `Relu`            | ✅ Done                        |                                  |
| `Shape`           | ✅ Done                        |                                  |
| `ReduceMean`      | ✅ Done                        |                                  |
| `Gather`          | ✅ Done                        |                                  |
| `Mul`             | ✅ Done                        |                                  |
| `Where`           | ✅ Done                        |                                  |
| `Add`             | ✅ Done                        |                                  |
| `Slice`           | ✅ Done                        |                                  |
| `ConstantOfShape` | ✅ Done                        |                                  |
| `Transpose`       | ✅ Done                        |                                  |
| `Range`           | ✅ Done                        |                                  |
| `Concat`          | ✅ Done                        |                                  |
| `Reshape`         | ✅ Done                        |                                  |
| `Constant`        | ✅ Done                        |                                  |

## Notes

- The `ConstantOfShape` definition change (in terms of attributes) should be considered in the converter.
- `ReduceMean` axes input should use `usize` type.
- Check how we handle inputs for `ScatterND`.
