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
        "name": "b",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "a+b",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "constant": true,
        "name": "const1",
        "dtype": "float",
        "shape": [1, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "(a+b)+const1",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "add"
      },
      "input": ["a", "b"],
      "output": ["a+b"]
    },
    {
      "info": {
        "type": "op",
        "name": "add"
      },
      "input": ["a+b", "const1"],
      "output": ["(a+b)+const1"]
    }
  ],
  "inputs": ["a", "b"],
  "outputs": ["(a+b)+const1"],
  "tensors": [
    {
        "name": "const1",
        "content": {
            "dtype": "float",
            "data": [3.0, 4.0]
        }
    }
  ]
}
"#;

#[test]
fn test_add_broadcast() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "a".to_string(),
            vec![1.0, 2.0, 3.0, 4.0].try_into().unwrap(),
        ),
        (
            "b".to_string(),
            vec![1.0, 2.0, 3.0, 4.0].try_into().unwrap(),
        ),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("(a+b)+const1").unwrap().try_into().unwrap();
    assert_eq!(data, vec![5.0, 8.0, 9.0, 12.0]);
}
