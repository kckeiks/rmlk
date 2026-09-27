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
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "sqrt(a)",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "sqrt"
      },
      "input": ["a"],
      "output": ["sqrt(a)"]
    }
  ],
  "inputs": ["a"],
  "outputs": ["sqrt(a)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [(
        "a".to_string(),
        vec![1.0, 4.0, 9.0, 5.0].try_into().unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("sqrt(a)").unwrap().try_into().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 2.236_068]);
}
