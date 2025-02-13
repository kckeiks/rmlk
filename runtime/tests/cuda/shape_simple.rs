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
        "shape": [3, 4, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "shape(data)",
        "dtype": "int64",
        "shape": [3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "shape"
      },
      "input": ["data"],
      "output": ["shape(data)"]
    }
  ],
  "inputs": ["data"],
  "outputs": ["shape(data)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> =
        [("data".to_string(), vec![0.0; 3 * 4 * 2].try_into().unwrap())].into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<i64> = output.remove("shape(data)").unwrap().try_into().unwrap();
    assert_eq!(data, vec![3, 4, 2]);
}
