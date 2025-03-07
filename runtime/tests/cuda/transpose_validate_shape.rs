use crate::common;
use rmlk_runtime::Value;
use std::collections::HashMap;

const GRAPH_DEFINITION: &str = r#"
{
  "nodes": [
    {
      "info": {
        "type": "value",
        "name": "input",
        "dtype": "float",
        "shape": [3, 4, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "transpose(input)",
        "dtype": "float",
        "shape": [4, 3, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "shape(transpose(input))",
        "dtype": "int64",
        "shape": [3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "transpose"
      },
      "input": ["input"],
      "output": ["transpose(input)"]
    },
    {
      "info": {
        "type": "op",
        "name": "shape"
      },
      "input": ["transpose(input)"],
      "output": ["shape(transpose(input))"]
    }
  ],
  "inputs": ["input"],
  "outputs": ["shape(transpose(input))"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [(
        "input".to_string(),
        vec![1.0; 3 * 4 * 2].try_into().unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<i64> = output
        .remove("shape(transpose(input))")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![2, 4, 3]);
}

const GRAPH_DEFINITION_WITH_ATTR: &str = r#"
{
  "nodes": [
    {
      "info": {
        "type": "value",
        "name": "input",
        "dtype": "float",
        "shape": [3, 4, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "transpose(input)",
        "dtype": "float",
        "shape": [4, 3, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "shape(transpose(input))",
        "dtype": "int64",
        "shape": [3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "transpose",
        "attributes": {
            "perm": {
                "type": "ints",
                "data": [1, 0, 2]
            }
        }
      },
      "input": ["input"],
      "output": ["transpose(input)"]
    },
    {
      "info": {
        "type": "op",
        "name": "shape"
      },
      "input": ["transpose(input)"],
      "output": ["shape(transpose(input))"]
    }
  ],
  "inputs": ["input"],
  "outputs": ["shape(transpose(input))"],
  "tensors": []
}
"#;

#[test]
fn test_run_with_attr() {
    let mut instance = common::build(GRAPH_DEFINITION_WITH_ATTR).build().unwrap();
    let input: HashMap<String, Value> = [(
        "input".to_string(),
        vec![1.0; 3 * 4 * 2].try_into().unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<i64> = output
        .remove("shape(transpose(input))")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![4, 3, 2]);
}
