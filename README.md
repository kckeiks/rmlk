# rmlk

Rusty Machine Learning Kit is a machine learning library.

Based on ONNX version: 1.13.1

## Todo Checklist

| Operator          | Status                  | Notes                                  |
|-------------------|------------------------|----------------------------------------|
| `Expand`          | 🛠️ In Progress         | Needs kernel, backend, and tests       |
| `Sin`             | 🛠️ In Progress         | Needs kernel, backend, and tests       |
| `Sigmoid`         | ✅ Done          |      |
| `Relu`            | ✅ Done          |      |
| `Cast`            | 🛠️ In Progress         | Needs backend and tests                |
| `Unsqueeze`       | ✅ Done                  |                                        |
| `Shape`           | ✅ Done                 |                                        |
| `ReduceMean`      | ✅ Done                 |                                        |
| `Pow`             | 🛠️ In Progress         | Needs kernel, backend, and tests       |
| `Gather`          | ✅ Done                 |                                        |
| `Mul`             | ✅ Done                 |                                        |
| `Trilu`           | ❓ Not Started          |                                        |
| `Where`           | ✅ Done                 |                                        |
| `Sqrt`            | 🛠️ In Progress         | Needs backend and tests                |
| `Add`             | ✅ Done                 |                                        |
| `Neg`             | 🛠️ In Progress         | Needs kernel, backend, and tests       |
| `Slice`           | ✅ Done             |                                        |
| `MatMul`          | ❓ Not Started          |                                        |
| `ScatterND`       | 🛠️ In Progress         | Needs backend and more tests           |
| `Equal`           | 🛠️ In Progress         | Needs kernel, backend, and tests       |
| `ConstantOfShape` | ✅ Done                 |                                        |
| `Transpose`       | ✅ Done                 |                                        |
| `Range`           | ✅ Done         |                                        |
| `Concat`          | ✅ Done           |                                        |
| `Div`             | 🛠️ In Progress         | Needs kernel, backend, and tests       |
| `Greater`         | 🛠️ In Progress         | Needs kernel, backend, and tests       |
| `Softmax`         | ❓ Not Started          |                                        |
| `Reshape`         | ✅ Done                 |                                        |
| `Cos`             | 🛠️ In Progress         | Needs kernel, backend, and tests       |
| `Sub`             | 🛠️ In Progress         | Needs kernel, backend, and tests       |
| `Constant`        | ✅ Done            |                                        |

## Notes

- The `ConstantOfShape` definition change (in terms of attributes) should be considered in the converter.
- `ReduceMean` axes input should use `usize` type.
- Check how we handle inputs for `ScatterND`.
