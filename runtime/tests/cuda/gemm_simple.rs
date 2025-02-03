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
        "shape": [3, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "b",
        "dtype": "float",
        "shape": [2, 4]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "a*b",
        "dtype": "float",
        "shape": [3, 4]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "gemm"
      },
      "input": ["a", "b"],
      "output": ["a*b"]
    }
  ],
  "inputs": ["a", "b"],
  "outputs": ["a*b"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "a".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].try_into().unwrap(),
        ),
        (
            "b".to_string(),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
                .try_into()
                .unwrap(),
        ),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("a*b").unwrap().try_into().unwrap();
    assert_eq!(
        data,
        vec![29.0, 32.0, 35.0, 38.0, 65.0, 72.0, 79.0, 86.0, 101.0, 112.0, 123.0, 134.0]
    );
}
