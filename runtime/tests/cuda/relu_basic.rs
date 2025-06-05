use crate::common;
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
        "shape": [1, 2, 2, 1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "relu(a)",
        "dtype": "float",
        "shape": [1, 2, 2, 1]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "relu"
      },
      "input": ["a"],
      "output": ["relu(a)"]
    }
  ],
  "inputs": ["a"],
  "outputs": ["relu(a)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [(
        "a".to_string(),
        vec![10.0, -10.0, 5.0, -5.0].try_into().unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("relu(a)").unwrap().try_into().unwrap();
    assert_eq!(data, vec![10.0, 0.0, 5.0, 0.0]);
}
