use crate::common;
use rmlk_runtime::Value;
use std::collections::HashMap;
use std::f32::consts::PI;

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
        "name": "sin(a)",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "sin"
      },
      "input": ["a"],
      "output": ["sin(a)"]
    }
  ],
  "inputs": ["a"],
  "outputs": ["sin(a)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [(
        "a".to_string(),
        vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0].try_into().unwrap(),
    )]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("sin(a)").unwrap().try_into().unwrap();
    let expected_output = vec![
        0.0_f32.sin(),
        (PI / 2.0).sin(),
        PI.sin(),
        (3.0 * PI / 2.0).sin(),
    ];
    assert_eq!(data, expected_output);
}
