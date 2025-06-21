// use crate::common;
// use rmlk_runtime::Value;
// use std::collections::HashMap;
//
// const GRAPH_DEFINITION: &str = r#"
// {
//   "nodes": [
//     {
//       "info": {
//         "type": "value",
//         "name": "a",
//         "dtype": "float",
//         "shape": [1, 4, 8]
//       }
//     },
//     {
//       "info": {
//         "type": "value",
//         "name": "b",
//         "dtype": "float",
//         "shape": [8, 6]
//       }
//     },
//     {
//       "info": {
//         "type": "value",
//         "name": "matmul(a,b)",
//         "dtype": "float",
//         "shape": [1, 4, 6]
//       }
//     },
//     {
//       "info": {
//         "type": "op",
//         "name": "matmul"
//       },
//       "input": ["a", "b"],
//       "output": ["matmul(a,b)"]
//     }
//   ],
//   "inputs": ["a", "b"],
//   "outputs": ["matmul(a,b)"],
//   "tensors": []
// }
// "#;
//
// #[test]
// fn test_run() {
//     let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
//     let a = build_a();
//     let b = build_b();
//
//     let input: HashMap<String, Value> = [
//         ("a".to_string(), a.clone().try_into().unwrap()),
//         ("b".to_string(), b.clone().try_into().unwrap()),
//     ]
//     .into();
//     let mut output = instance.run(input).unwrap();
//     let expected = matmul_reference(&a, &b);
//     let data: Vec<f32> = output.remove("matmul(a,b)").unwrap().try_into().unwrap();
//     assert_eq!(data, expected);
// }
//
// const M: usize = 1; // leading batch dim (ignored in flat layout)
//
// const N_MID: usize = 4; // second dim
// const K: usize = 8; // shared dimension
// const N: usize = 6; // output / B's columns
//
// // A : (1, 27, 3072)  =>  27 * 3072 elements
// fn build_a() -> Vec<f32> {
//     (0..N_MID * K).map(|i| i as f32).collect()
// }
//
// // B : (3072, 1024)
// fn build_b() -> Vec<f32> {
//     (0..K)
//         .flat_map(|r| (0..N).map(move |c| r as f32 + c as f32 * 0.001))
//         .collect()
// }
//
// fn matmul_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
//     let mut c = vec![0f32; N_MID * N];
//
//     for i in 0..N_MID {
//         for k in 0..K {
//             let a_val = a[i * K + k];
//             let b_row = &b[k * N..(k + 1) * N];
//             for j in 0..N {
//                 c[i * N + j] += a_val * b_row[j];
//             }
//         }
//     }
//     c
// }
