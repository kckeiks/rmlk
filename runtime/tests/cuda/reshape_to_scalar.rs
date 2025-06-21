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
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "b",
        "dtype": "int64",
        "shape": []
      }
    },
    {
      "info": {
        "type": "value",
        "name": "reshape(a, b)",
        "dtype": "float",
        "shape": []
      }
    },
        {
      "info": {
        "type": "value",
        "name": "shape(reshape(a, b))",
        "dtype": "int64",
        "shape": [0]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "reshape"
      },
      "input": ["a", "b"],
      "output": ["reshape(a, b)"]
    },{
      "info": {
        "type": "op",
        "name": "shape"
      },
      "input": ["reshape(a, b)"],
      "output": ["shape(reshape(a, b))"]
    }
  ],
  "inputs": ["a", "b"],
  "outputs": ["shape(reshape(a, b))"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        ("a".to_string(), vec![2].try_into().unwrap()),
        ("b".to_string(), Vec::<i64>::new().try_into().unwrap()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<i64> = output
        .remove("shape(reshape(a, b))")
        .unwrap()
        .try_into()
        .unwrap();
    assert!(data.is_empty());
}
