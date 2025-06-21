use crate::common;
use rmlk_runtime::Value;
use std::collections::HashMap;

const GRAPH_DEFINITION: &str = r#"
{
  "nodes": [
    {
      "info": {
        "type": "value",
        "name": "condition",
        "dtype": "bool",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "x",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "y",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "x{where(condition)}y",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "where"
      },
      "input": ["condition", "x", "y"],
      "output": ["x{where(condition)}y"]
    }
  ],
  "inputs": ["condition", "x", "y"],
  "outputs": ["x{where(condition)}y"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "x".to_string(),
            vec![11.0, 22.0, 33.0, 44.0].try_into().unwrap(),
        ),
        (
            "y".to_string(),
            vec![1.0, 2.0, 3.0, 4.0].try_into().unwrap(),
        ),
        (
            "condition".to_string(),
            vec![true, false, true, false].try_into().unwrap(),
        ),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("x{where(condition)}y")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![11.0, 2.0, 33.0, 4.0]);
}
