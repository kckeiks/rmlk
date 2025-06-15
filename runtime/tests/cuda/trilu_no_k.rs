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
        "shape": [2, 3, 3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "trilu(input)",
        "dtype": "float",
        "shape": [2, 3, 3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "trilu",
        "attributes": {
            "upper": {
                "type": "int",
                "data": 0
            }
        }
      },
      "input": ["input"],
      "output": ["trilu(input)"]
    }
  ],
  "inputs": ["input"],
  "outputs": ["trilu(input)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [(
        "input".to_string(),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0,
        ]
        .try_into()
        .unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("trilu(input)").unwrap().try_into().unwrap();
    let expected = vec![
        1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, 13.0, 14.0, 0.0, 16.0, 17.0,
        18.0,
    ];
    assert_eq!(data, expected);
}
