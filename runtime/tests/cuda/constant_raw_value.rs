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
        "dtype": "int",
        "shape": [3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "constant",
        "attributes": {
            "value": {
                "type": "tensor",
                "data": {
                  "dims": [3],
                  "data_type": "Int32",
                  "float_data": [],
                  "int32_data": [],
                  "string_data": [],
                  "int64_data": [],
                  "double_data": [],
                  "uint64_data": [],
                  "raw_data": [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]
                }
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
    let data: Vec<i32> = output.remove("output").unwrap().try_into().unwrap();
    assert_eq!(data, vec![1, 0, 2]);
}
