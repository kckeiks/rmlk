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
        "shape": [1, 1, 1, 4]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "softmax(a)",
        "dtype": "float",
        "shape": [1, 1, 1, 4]
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
        vec![0.0, 1.0, 2.0, 3.0].try_into().unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("softmax(a)").unwrap().try_into().unwrap();
    assert_eq!(data, vec![0.032058604, 0.087144315, 0.23688282, 0.6439143]);
}
