use crate::common;
use rmlk_runtime::Value;
use std::collections::HashMap;

const GRAPH_DEFINITION: &str = r#"
{
  "nodes": [
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
        "dtype": "int",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "x^y",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "pow"
      },
      "input": ["x", "y"],
      "output": ["x^y"]
    }
  ],
  "inputs": ["x", "y"],
  "outputs": ["x^y"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "x".to_string(),
            vec![1.0, 2.0, 3.0, 4.0].try_into().unwrap(),
        ),
        (
            "y".to_string(),
            vec![2i32, 2, 3, 2].try_into().unwrap(),
        ),
    ]
        .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("x^y").unwrap().try_into().unwrap();
    assert_eq!(data, vec![1.0, 4.0, 27.0, 16.0]);
}
