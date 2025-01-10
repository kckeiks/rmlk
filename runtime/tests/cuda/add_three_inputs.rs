use crate::common;
use rmlk_runtime::Value;
use std::collections::HashMap;

const TEST_GRAPH_DEFINITION: &str = r#"
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
        "name": "c",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "(a+b)+c",
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
      "input": ["a+b", "c"],
      "output": ["(a+b)+c"]
    }
  ],
  "inputs": ["a", "b", "c"],
  "outputs": ["(a+b)+c"],
  "tensors": []
}
"#;

#[test]
fn test_add_more_operands() {
    let mut instance = common::build(TEST_GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "a".to_string(),
            vec![1.0, 2.0, 3.0, 4.0].try_into().unwrap(),
        ),
        (
            "b".to_string(),
            vec![1.0, 2.0, 3.0, 4.0].try_into().unwrap(),
        ),
        (
            "c".to_string(),
            vec![1.0, 2.0, 3.0, 4.0].try_into().unwrap(),
        ),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("(a+b)+c").unwrap().try_into().unwrap();
    assert_eq!(data, vec![3.0, 6.0, 9.0, 12.0]);
}
