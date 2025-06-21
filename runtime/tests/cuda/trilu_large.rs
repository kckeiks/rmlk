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
        "shape": [3, 3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "k",
        "dtype": "int64",
        "shape": []
      }
    },
    {
      "info": {
        "type": "value",
        "name": "trilu(input)",
        "dtype": "float",
        "shape": [3, 3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "trilu",
        "attributes": {
            "upper": {
                "type": "int",
                "data": 1
            }
        }
      },
      "input": ["input", "k"],
      "output": ["trilu(input)"]
    }
  ],
  "inputs": ["input", "k"],
  "outputs": ["trilu(input)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "input".to_string(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                .try_into()
                .unwrap(),
        ),
        ("k".to_string(), vec![1i64].try_into().unwrap()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("trilu(input)").unwrap().try_into().unwrap();
    let expected = vec![0.0, 1.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0];
    assert_eq!(data, expected);
}
