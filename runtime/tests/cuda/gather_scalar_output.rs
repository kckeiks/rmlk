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
        "shape": [4]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "indices",
        "dtype": "float",
        "shape": []
      }
    },
    {
      "info": {
        "type": "value",
        "name": "gather(data, indices)",
        "dtype": "float",
        "shape": []
      }
    },
    {
      "info": {
        "type": "op",
        "name": "gather"
      },
      "input": ["data", "indices"],
      "output": ["gather(data, indices)"]
    }
  ],
  "inputs": ["data", "indices"],
  "outputs": ["gather(data, indices)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "data".to_string(),
            vec![10.0, 20.0, 30.0, 40.0].try_into().unwrap(),
        ),
        ("indices".to_string(), vec![2i32].try_into().unwrap()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("gather(data, indices)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![30.0]);
}
