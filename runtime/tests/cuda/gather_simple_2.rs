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
        "shape": [3, 4]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "indices",
        "dtype": "float",
        "shape": [2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "gather(data, indices)",
        "dtype": "float",
        "shape": [2, 4]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "gather",
        "attributes": {
            "axis": {
                "type": "int",
                "data": 1
            }
        }
      },
      "input": ["data", "indices"],
      "output": ["gather(data, indices)"]
    }
  ],
  "inputs": ["data", "indices"],
  "outputs": ["gather(data, indices)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "data".to_string(),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]
            .try_into()
            .unwrap(),
        ),
        ("indices".to_string(), vec![1i32, 3i32].try_into().unwrap()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("gather(data, indices)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0,]);
}
