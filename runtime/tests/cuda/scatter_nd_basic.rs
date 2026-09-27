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
        "shape": [4, 4, 3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "indices",
        "dtype": "int64",
        "shape": [3, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "updates",
        "dtype": "float",
        "shape": [3, 3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "scatternd(a)",
        "dtype": "float",
        "shape": [4, 4, 3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "scatternd"
      },
      "input": ["a", "indices", "updates"],
      "output": ["scatternd(a)"]
    }
  ],
  "inputs": ["a", "indices", "updates"],
  "outputs": ["scatternd(a)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let data: Vec<f32> = (0..48).map(|x| x as f32).collect();
    let indices: Vec<i64> = vec![0, 1, 2, 0, 3, 2];
    let updates: Vec<f32> = vec![10.1, 10.2, 10.3, 20.1, 20.2, 20.3, 30.1, 30.2, 30.3];
    let input: HashMap<String, Value> = [
        ("a".to_string(), data.into()),
        ("indices".to_string(), indices.into()),
        ("updates".to_string(), updates.into()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("scatternd(a)").unwrap().try_into().unwrap();
    let expected: Vec<f32> = vec![
        0.0, 1.0, 2.0, 10.1, 10.2, 10.3, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 20.1, 20.2, 20.3, 27.0, 28.0, 29.0, 30.0,
        31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 30.1, 30.2, 30.3, 45.0,
        46.0, 47.0,
    ];
    assert_eq!(data, expected);
}
