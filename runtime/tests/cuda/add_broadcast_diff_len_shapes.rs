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
        "shape": [2, 4, 1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "b",
        "dtype": "float",
        "shape": [4, 3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "a+b",
        "dtype": "float"
      }
    },
    {
      "info": {
        "type": "value",
        "constant": true,
        "name": "const1",
        "dtype": "float",
        "shape": [3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "(a+b)+const1",
        "dtype": "float",
        "shape": [2, 4, 3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "add"
      },
      "input": ["a", "b"],
      "output": ["a+b"]
    },
    {
      "info": {
        "type": "op",
        "name": "add"
      },
      "input": ["a+b", "const1"],
      "output": ["(a+b)+const1"]
    }
  ],
  "inputs": ["a", "b"],
  "outputs": ["(a+b)+const1"],
  "tensors": [
    {
        "name": "const1",
        "content": {
            "dtype": "float",
            "data": [100.0, 200.0, 300.0]
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
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
                .try_into()
                .unwrap(),
        ),
        (
            "b".to_string(),
            vec![
                1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1,
            ]
            .try_into()
            .unwrap(),
        ),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output.remove("(a+b)+const1").unwrap().try_into().unwrap();
    assert_eq!(
        data,
        vec![
            111.1, 212.1, 313.1, 124.1, 225.1, 326.1, 137.1, 238.1, 339.1, 150.1, 251.1, 352.1,
            151.1, 252.1, 353.1, 164.1, 265.1, 366.1, 177.1, 278.1, 379.1, 190.1, 291.1, 392.1,
        ]
    );
}
