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
        "shape": [2, 2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "softmax(a)",
        "dtype": "float",
        "shape": [2, 2, 2]
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
        vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]
            .try_into()
            .unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("softmax(a)").unwrap().try_into().unwrap();
    assert_eq!(
        data,
        vec![
            0.2689414, 0.7310586, 0.26894143, 0.7310586, 0.26894143, 0.7310586, 0.2689414,
            0.73105854
        ]
    );
}
