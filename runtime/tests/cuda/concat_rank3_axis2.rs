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
        "shape": [2, 3, 4]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "b",
        "dtype": "float",
        "shape": [2, 3, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "concat(a, b)",
        "dtype": "float",
        "shape": [2, 3, 6]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "concat",
        "attributes": {
            "axis": {
                "type": "int",
                "data": 2
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
    let a = (1..=24).map(|x| x as f32).collect::<Vec<_>>();
    let b = (100..=111).map(|x| x as f32).collect::<Vec<_>>();
    let input: HashMap<String, Value> = [
        ("a".to_string(), a.clone().into()),
        ("b".to_string(), b.clone().into()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("concat(a, b)").unwrap().try_into().unwrap();
    let expected = vec![
        // batch 0
        1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 5.0, 6.0, 7.0, 8.0, 102.0, 103.0, 9.0, 10.0, 11.0, 12.0,
        104.0, 105.0, // batch 1
        13.0, 14.0, 15.0, 16.0, 106.0, 107.0, 17.0, 18.0, 19.0, 20.0, 108.0, 109.0, 21.0, 22.0,
        23.0, 24.0, 110.0, 111.0,
    ];
    assert_eq!(data, expected);
}
