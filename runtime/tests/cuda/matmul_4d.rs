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
        "shape": [1, 24, 6, 6]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "b",
        "dtype": "float",
        "shape": [1, 24, 6, 128]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "matmul(a,b)",
        "dtype": "float",
        "shape": [1, 24, 6, 128]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "shape(matmul(a,b))",
        "dtype": "int64",
        "shape": [4]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "matmul"
      },
      "input": ["a", "b"],
      "output": ["matmul(a,b)"]
    },
    {
      "info": {
        "type": "op",
        "name": "shape"
      },
      "input": ["matmul(a,b)"],
      "output": ["shape(matmul(a,b))"]
    }
  ],
  "inputs": ["a", "b"],
  "outputs": ["shape(matmul(a,b))"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "a".to_string(),
            vec![1.0; 1 * 24 * 6 * 6].try_into().unwrap(),
        ),
        (
            "b".to_string(),
            vec![13.0; 1 * 24 * 6 * 128].try_into().unwrap(),
        ),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<i64> = output
        .remove("shape(matmul(a,b))")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![1, 24, 6, 128]);
}
