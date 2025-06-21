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
        "shape": [2, 3]
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
        "name": "reducemean(input, axes)",
        "dtype": "float",
        "shape": [1, 3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "reducemean"
      },
      "input": ["input", "axes"],
      "output": ["reducemean(input, axes)"]
    }
  ],
  "inputs": ["input", "axes"],
  "outputs": ["reducemean(input, axes)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "input".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].try_into().unwrap(),
        ),
        ("axes".to_string(), vec![0i64].try_into().unwrap()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("reducemean(input, axes)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![2.5, 3.5, 4.5]);
}
