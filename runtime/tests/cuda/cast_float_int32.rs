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
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "cast(a)",
        "dtype": "int",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "cast",
        "attributes": {
            "to": {
                "type": "int",
                "data": 6
            }
        }
      },
      "input": ["a"],
      "output": ["cast(a)"]
    }
  ],
  "inputs": ["a"],
  "outputs": ["cast(a)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [(
        "a".to_string(),
        vec![1.0, 2.0, 3.0, 4.0].try_into().unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<i32> = output.remove("cast(a)").unwrap().try_into().unwrap();
    assert_eq!(data, vec![1, 2, 3, 4]);
}
