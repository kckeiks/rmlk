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
        "shape": []
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
        "name": "a+b",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "add"
      },
      "input": ["a", "b"],
      "output": ["a+b"]
    }
  ],
  "inputs": ["a", "b"],
  "outputs": ["a+b"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        ("a".to_string(), vec![4.0].try_into().unwrap()),
        (
            "b".to_string(),
            vec![2.0, 1.0, 10.0, 20.0].try_into().unwrap(),
        ),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("a+b").unwrap().try_into().unwrap();
    assert_eq!(data, vec![6.0, 5.0, 14.0, 24.0]);
}
