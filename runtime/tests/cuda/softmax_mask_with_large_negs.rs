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
        "shape": [2, 1, 1, 3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "softmax(a)",
        "dtype": "float",
        "shape": [2, 1, 1, 3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "softmax",
        "attributes": {
            "axis": {
                "type": "int",
                "data": -1
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
        vec![0.0, -10_000.0, 0.5, -7.0, -7.0, -10_000.0]
            .try_into()
            .unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("softmax(a)").unwrap().try_into().unwrap();
    assert_eq!(data, vec![0.377_540_68, 0.0, 0.622_459_35, 0.5, 0.5, 0.0,]);
}
