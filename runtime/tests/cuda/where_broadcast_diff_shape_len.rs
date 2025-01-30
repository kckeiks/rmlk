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
        "shape": [3, 2, 1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "b",
        "dtype": "float",
        "shape": [1, 6]
      }
    },
    {
      "info": {
        "type": "value",
        "constant": true,
        "name": "const1",
        "dtype": "float",
        "shape": [2, 6]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "where(const1, a, b)",
        "dtype": "float",
        "shape": [3, 2, 6]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "where"
      },
      "input": ["a", "b", "const1"],
      "output": ["where(const1, a, b)"]
    }
  ],
  "inputs": ["a", "b"],
  "outputs": ["where(const1, a, b)"],
  "tensors": [
    {
        "name": "const1",
        "content": {
            "dtype": "float",
            "data": [0.0, 1.0, 3.0, 0.0, 5.6, 22.44, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        }
    }
  ]
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "a".to_string(),
            vec![
                0.40744543, 0.09136569, 0.72850689, 0.33760798, 0.80262239, 0.41559307,
            ]
            .try_into()
            .unwrap(),
        ),
        (
            "b".to_string(),
            vec![
                0.69894743, 0.53463184, 0.80940321, 0.86458025, 0.34860543, 0.67579115,
            ]
            .try_into()
            .unwrap(),
        ),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("where(const1, a, b)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(
        data,
        vec![
            0.69894743, 0.40744543, 0.40744543, 0.86458025, 0.40744543, 0.40744543, 0.69894743,
            0.53463184, 0.80940321, 0.86458025, 0.09136569, 0.67579115, 0.69894743, 0.72850689,
            0.72850689, 0.86458025, 0.72850689, 0.72850689, 0.69894743, 0.53463184, 0.80940321,
            0.86458025, 0.33760798, 0.67579115, 0.69894743, 0.80262239, 0.80262239, 0.86458025,
            0.80262239, 0.80262239, 0.69894743, 0.53463184, 0.80940321, 0.86458025, 0.41559307,
            0.67579115
        ]
    );
}
