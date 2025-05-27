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
        "shape": [2, 3, 2, 4]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "b",
        "dtype": "float",
        "shape": [2, 3, 2, 1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "concat(a, b)",
        "dtype": "float",
        "shape": [2, 3, 2, 5]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "concat",
        "attributes": {
            "axis": {
                "type": "int",
                "data": 3
            }
        }
      },
      "input": ["a", "b"],
      "output": ["concat(a, b)"]
    }
  ],
  "inputs": ["a", "b"],
  "outputs": ["concat(a, b)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let a = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
        47.0, 48.0,
    ];
    let b = vec![
        101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0,
    ];
    let input: HashMap<String, Value> = [
        ("a".to_string(), a.clone().try_into().unwrap()),
        ("b".to_string(), b.clone().try_into().unwrap()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("concat(a, b)").unwrap().try_into().unwrap();
    let expected = vec![
        1.0, 2.0, 3.0, 4.0, 101.0, 5.0, 6.0, 7.0, 8.0, 102.0, 9.0, 10.0, 11.0, 12.0, 103.0, 13.0,
        14.0, 15.0, 16.0, 104.0, 17.0, 18.0, 19.0, 20.0, 105.0, 21.0, 22.0, 23.0, 24.0, 106.0,
        25.0, 26.0, 27.0, 28.0, 107.0, 29.0, 30.0, 31.0, 32.0, 108.0, 33.0, 34.0, 35.0, 36.0,
        109.0, 37.0, 38.0, 39.0, 40.0, 110.0, 41.0, 42.0, 43.0, 44.0, 111.0, 45.0, 46.0, 47.0,
        48.0, 112.0,
    ];
    assert_eq!(data, expected);
}
