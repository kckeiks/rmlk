use crate::common;
use rmlk_runtime::Value;
use std::collections::HashMap;

const GRAPH_DEFINITION: &str = r#"
{
  "nodes": [
    {
      "info": {
        "type": "value",
        "name": "output",
        "dtype": "float",
        "shape": []
      }
    },
    {
      "info": {
        "type": "op",
        "name": "constant",
        "attributes": {
            "value_float": {
                "type": "float",
                "data": 69.0
            }
        }
      },
      "input": [],
      "output": ["output"]
    }
  ],
  "inputs": [],
  "outputs": ["output"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = HashMap::new();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("output").unwrap().try_into().unwrap();
    assert_eq!(data, vec![69.0]);
}
