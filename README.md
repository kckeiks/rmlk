# rmlk

Rusty Machine Learning Kit is a machine learning library.

Based on ONNX version: 1.13.1

## Todo Checklist

| Operator          | Status              | Notes                            |
|-------------------|---------------------|----------------------------------|
| `Expand`          | ❓ Not Started       |                                  |
| `Trilu`           | ❓ Not Started       |                                  |
| `Softmax`         | ❓ Not Started       |                                  |
| `MatMul`          | ❓ Not Started       |                                  |
| `ScatterND`       | 🛠️ In Progress     | Needs backend and more tests     |
| `Greater`         | 🛠️ In Progress     | Needs kernel, backend, and tests |
| `Equal`           | 🛠️ In Progress     | Needs kernel, backend, and tests |
| `Sin`             | 🛠️ In Progress     | Needs kernel, backend, and tests |
| `Cast`            | 🛠️ In Progress     | Needs backend and tests          |
| `Pow`             | 🛠️ In Progress     | Needs kernel, backend, and tests |
| `Sqrt`            | 🛠️ In Progress     | Needs backend and tests          |
| `Neg`             | 🛠️ In Progress     | Needs kernel, backend, and tests |
| `Div`             | 🛠️ In Progress     | Needs kernel, backend, and tests |
| `Cos`             | 🛠️ In Progress     | Needs kernel, backend, and tests |
| `Sub`             | 🛠️ In Progress     | Needs kernel, backend, and tests |
| `Unsqueeze`       | ✅ Done              |                                  |
| `Sigmoid`         | ✅ Done              |                                  |
| `Relu`            | ✅ Done              |                                  |
| `Shape`           | ✅ Done              |                                  |
| `ReduceMean`      | ✅ Done              |                                  |
| `Gather`          | ✅ Done              |                                  |
| `Mul`             | ✅ Done              |                                  |
| `Where`           | ✅ Done              |                                  |
| `Add`             | ✅ Done              |                                  |
| `Slice`           | ✅ Done              |                                  |
| `ConstantOfShape` | ✅ Done              |                                  |
| `Transpose`       | ✅ Done              |                                  |
| `Range`           | ✅ Done              |                                  |
| `Concat`          | ✅ Done              |                                  |
| `Reshape`         | ✅ Done              |                                  |
| `Constant`        | ✅ Done              |                                  |

## Notes

- The `ConstantOfShape` definition change (in terms of attributes) should be considered in the converter.
- `ReduceMean` axes input should use `usize` type.
- Check how we handle inputs for `ScatterND`.
