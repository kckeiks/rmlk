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
        "shape": [3, 2, 3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "starts",
        "dtype": "int",
        "shape": [2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "ends",
        "dtype": "int",
        "shape": [2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "axes",
        "dtype": "int",
        "shape": [2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "steps",
        "dtype": "int",
        "shape": [2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "slice(data, starts, ends, axes, steps)",
        "dtype": "float",
        "shape": [3, 2, 2]
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
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.03, 14.0, 15.0,
                16.0, 17.0, 18.0,
            ]
            .try_into()
            .unwrap(),
        ),
        ("starts".to_string(), vec![0i32, 1].try_into().unwrap()),
        ("ends".to_string(), vec![3i32, 3].try_into().unwrap()),
        ("axes".to_string(), vec![0i32, 2].try_into().unwrap()),
        ("steps".to_string(), vec![1i32, 1].try_into().unwrap()),
    ]
    .into();
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("slice(data, starts, ends, axes, steps)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(
        data,
        vec![2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0, 14.0, 15.0, 17.0, 18.0]
    );
}
