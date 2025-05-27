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
        "shape": [2, 3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "b",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "concat(a, b)",
        "dtype": "float",
        "shape": [2, 5]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "concat",
        "attributes": {
            "axis": {
                "type": "int",
                "data": 1
            }
        }
      },
      "input": ["a", "b"],
      "output": ["concat(a, b)"]
    }
  ],
  "inputs": ["a", "b"],
  "outputs": ["concat(a, b)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![100.0, 101.0, 200.0, 201.0];
    let input: HashMap<String, Value> = [
        ("a".to_string(), a.clone().try_into().unwrap()),
        ("b".to_string(), b.clone().try_into().unwrap()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("concat(a, b)").unwrap().try_into().unwrap();
    assert_eq!(
        data,
        vec![1.0, 2.0, 3.0, 100.0, 101.0, 4.0, 5.0, 6.0, 200.0, 201.0,]
    );
}
