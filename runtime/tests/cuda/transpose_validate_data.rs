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
        "shape": [2, 4, 3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "transpose"
      },
      "input": ["input"],
      "output": ["transpose(input)"]
    }
  ],
  "inputs": ["input"],
  "outputs": ["transpose(input)"],
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
    let data: Vec<f32> = output
        .remove("transpose(input)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![1.0; 3 * 4 * 2]);
}
