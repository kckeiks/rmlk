use crate::common;
use rmlk_runtime::Value;
use std::collections::HashMap;

const GRAPH_DEFINITION: &str = r#"
{
  "nodes": [
    {
      "info": {
        "type": "value",
        "name": "data",
        "dtype": "float",
        "shape": [2, 2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "starts",
        "dtype": "int",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "ends",
        "dtype": "int",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "axes",
        "dtype": "int",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "steps",
        "dtype": "int",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "slice(data, starts, ends, axes, steps)",
        "dtype": "float",
        "shape": [2, 0, 2]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "slice"
      },
      "input": ["data", "starts", "ends", "axes", "steps"],
      "output": ["slice(data, starts, ends, axes, steps)"]
    }
  ],
  "inputs": ["data", "starts", "ends", "axes", "steps"],
  "outputs": ["slice(data, starts, ends, axes, steps)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let input: HashMap<String, Value> = [
        (
            "data".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                .try_into()
                .unwrap(),
        ),
        ("starts".to_string(), vec![1i32].try_into().unwrap()),
        ("ends".to_string(), vec![0i32].try_into().unwrap()),
        ("axes".to_string(), vec![1i32].try_into().unwrap()),
        ("steps".to_string(), vec![1i32].try_into().unwrap()),
    ]
    .into();
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("slice(data, starts, ends, axes, steps)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, Vec::<f32>::new());
}
