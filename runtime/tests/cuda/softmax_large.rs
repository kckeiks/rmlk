use crate::common;
use approx::abs_diff_eq;
use rmlk_runtime::Value;
use std::collections::HashMap;

const GRAPH_DEFINITION: &str = r#"
{
  "nodes": [
    {
      "info": {
        "type": "value",
        "name": "a",
        "dtype": "float",
        "shape": [1, 1, 1, 4096]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "softmax(a)",
        "dtype": "float",
        "shape": [1, 1, 1, 4096]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "softmax",
        "attributes": {
            "axis": {
                "type": "int",
                "data": -1
            }
        }
      },
      "input": ["a"],
      "output": ["softmax(a)"]
    }
  ],
  "inputs": ["a"],
  "outputs": ["softmax(a)"],
  "tensors": []
}
"#;

fn make_input() -> Vec<f32> {
    (0..4096).map(|i| ((i % 79) as f32) * 0.01 - 3.0).collect()
}

fn softmax_ref(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input = make_input();
    let expected = softmax_ref(&input);

    let input: HashMap<String, Value> = [("a".to_string(), input.into())].into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("softmax(a)").unwrap().try_into().unwrap();
    assert!(abs_diff_eq!(
        expected.as_slice(),
        data.as_slice(),
        epsilon = 1e-5
    ));
}
