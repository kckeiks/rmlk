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
        "name": "reducemean(input)",
        "dtype": "float",
        "shape": []
      }
    },
    {
      "info": {
        "type": "op",
        "name": "reducemean"
      },
      "input": ["input"],
      "output": ["reducemean(input)"]
    }
  ],
  "inputs": ["input"],
  "outputs": ["reducemean(input)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> =
        [("input".to_string(), vec![5.6].try_into().unwrap())].into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("reducemean(input)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![5.6]);
}
