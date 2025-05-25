use crate::common;
use rmlk_runtime::Value;
use std::collections::HashMap;

const GRAPH_DEFINITION: &str = r#"
{
  "nodes": [
    {
      "info": {
        "type": "value",
        "name": "data",
        "dtype": "float",
        "shape": [3, 4, 5]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "axes",
        "dtype": "int64",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "unsqueeze(axes, value)",
        "dtype": "float"
      }
    },
    {
      "info": {
        "type": "value",
        "name": "shape(unsqueeze(axes, value))",
        "dtype": "int64",
        "shape": [4]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "unsqueeze"
      },
      "input": ["data", "axes"],
      "output": ["unsqueeze(axes, value)"]
    },
    {
      "info": {
        "type": "op",
        "name": "shape"
      },
      "input": ["unsqueeze(axes, value)"],
      "output": ["shape(unsqueeze(axes, value))"]
    }
  ],
  "inputs": ["data", "axes"],
  "outputs": ["shape(unsqueeze(axes, value))"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        ("data".to_string(), vec![0.0; 3 * 4 * 5].try_into().unwrap()),
        ("axes".to_string(), vec![3i64].try_into().unwrap()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<i64> = output
        .remove("shape(unsqueeze(axes, value))")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![3, 4, 5, 1]);
}
