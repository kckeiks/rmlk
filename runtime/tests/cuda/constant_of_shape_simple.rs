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
        "dtype": "int64",
        "shape": [2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "constantofshape(input)",
        "dtype": "float",
        "shape": [3, 4]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "constantofshape"
      },
      "input": ["input"],
      "output": ["constantofshape(input)"]
    }
  ],
  "inputs": ["input"],
  "outputs": ["constantofshape(input)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> =
        [("input".to_string(), vec![3i64, 4i64].try_into().unwrap())].into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("constantofshape(input)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![0.0; 3 * 4]);
}
