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
        "shape": [2, 1, 3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "shape",
        "dtype": "float",
        "shape": [2, 4, 3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "expand(input)",
        "dtype": "float",
        "shape": [2, 4, 3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "expand"
      },
      "input": ["input", "shape"],
      "output": ["expand(input)"]
    }
  ],
  "inputs": ["input", "shape"],
  "outputs": ["expand(input)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "input".to_string(),
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ].try_into().unwrap(),
        ),
        (
            "shape".to_string(),
            vec![2i64, 4, 3].try_into().unwrap(),
        ),
    ]
        .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("expand(input)").unwrap().try_into().unwrap();
    let expected: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        4.0, 5.0, 6.0,
        4.0, 5.0, 6.0,
        4.0, 5.0, 6.0,
    ];
    assert_eq!(data, expected);
}
