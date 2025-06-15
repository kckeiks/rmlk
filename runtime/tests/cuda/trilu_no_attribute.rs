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
        "shape": [1, 3, 4]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "k",
        "dtype": "float",
        "shape": []
      }
    },
    {
      "info": {
        "type": "value",
        "name": "trilu(input)",
        "dtype": "float",
        "shape": [1, 3, 4]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "trilu"
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
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]
            .try_into()
            .unwrap(),
        ),
        ("k".to_string(), vec![1i64].try_into().unwrap()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("trilu(input)").unwrap().try_into().unwrap();
    let expected = vec![0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 12.0];
    assert_eq!(data, expected);
}
