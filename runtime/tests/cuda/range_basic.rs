use crate::common;
use rmlk_runtime::Value;
use std::collections::HashMap;

const GRAPH_DEFINITION: &str = r#"
{
  "nodes": [
    {
      "info": {
        "type": "value",
        "name": "start",
        "dtype": "int",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "limit",
        "dtype": "int",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "delta",
        "dtype": "int",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "range(start, limit, delta)",
        "dtype": "int",
        "shape": [5]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "range"
      },
      "input": ["start", "limit", "delta"],
      "output": ["range(start, limit, delta)"]
    }
  ],
  "inputs": ["start", "limit", "delta"],
  "outputs": ["range(start, limit, delta)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> =
        [
            ("start".to_string(), vec![0i32].try_into().unwrap()),
            ("limit".to_string(), vec![5i32].try_into().unwrap()),
            ("delta".to_string(), vec![1i32].try_into().unwrap()),
        ].into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<i32> = output.remove("range(start, limit, delta)").unwrap().try_into().unwrap();
    assert_eq!(data, vec![0, 1, 2, 3, 4]);
}
