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
        "shape": []
      }
    },
    {
      "info": {
        "type": "value",
        "name": "reducemean(input, axes)",
        "dtype": "float",
        "shape": []
      }
    },
    {
      "info": {
        "type": "op",
        "name": "reducemean",
        "attributes": {
            "noop_with_empty_axes": {
                "type": "int",
                "data": 1
            }
        }
      },
      "input": ["input"],
      "output": ["reducemean(input, axes)"]
    }
  ],
  "inputs": ["input"],
  "outputs": ["reducemean(input, axes)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> =
        [("input".to_string(), vec![4.6].try_into().unwrap())].into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("reducemean(input, axes)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![4.6]);
}
