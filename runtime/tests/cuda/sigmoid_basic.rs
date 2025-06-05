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
        "shape": [2, 2, 1, 1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "sigmoid(a)",
        "dtype": "float",
        "shape": [2, 2, 1, 1]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "sigmoid"
      },
      "input": ["a"],
      "output": ["sigmoid(a)"]
    }
  ],
  "inputs": ["a"],
  "outputs": ["sigmoid(a)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [(
        "a".to_string(),
        vec![0.5, -0.5, 2.0, -2.0].try_into().unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("sigmoid(a)").unwrap().try_into().unwrap();
    assert_eq!(data, vec![0.62245935, 0.37754068, 0.880797, 0.11920292]);
}
