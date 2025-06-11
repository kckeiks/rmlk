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
        "shape": [1, 3, 2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "softmax(a)",
        "dtype": "float",
        "shape": [1, 3, 2, 2]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "softmax",
        "attributes": {
            "axis": {
                "type": "int",
                "data": 1
            }
        }
      },
      "input": ["a"],
      "output": ["softmax(a)"]
    }
  ],
  "inputs": ["a"],
  "outputs": ["softmax(a)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [(
        "a".to_string(),
        vec![1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]
            .try_into()
            .unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("softmax(a)").unwrap().try_into().unwrap();
    assert_eq!(
        data,
        vec![
            0.24472846,
            0.46831053,
            0.7053845,
            0.86681336,
            0.66524094,
            0.46831053,
            0.25949645,
            0.117310435,
            0.09003057,
            0.06337894,
            0.035119027,
            0.015876241
        ]
    );
}
