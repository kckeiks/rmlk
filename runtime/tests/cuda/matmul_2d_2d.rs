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
        "shape": [3, 4]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "matmul(a,b)",
        "dtype": "float",
        "shape": [2, 4]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "matmul"
      },
      "input": ["a", "b"],
      "output": ["matmul(a,b)"]
    }
  ],
  "inputs": ["a", "b"],
  "outputs": ["matmul(a,b)"],
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
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ]
            .try_into()
            .unwrap(),
        ),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("matmul(a,b)").unwrap().try_into().unwrap();
    assert_eq!(
        data,
        vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
    );
}
