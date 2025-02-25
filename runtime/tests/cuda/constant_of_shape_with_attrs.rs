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
        "dtype": "int64",
        "shape": [2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "b",
        "dtype": "int64",
        "shape": [2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "a+b",
        "dtype": "int64",
        "shape": [2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "constantofshape(a+b)",
        "shape": [2]
      }
    },
    {
      "info": {
        "type": "value",
        "constant": true,
        "name": "const1",
        "dtype": "float",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "constantofshape(a+b)+const1",
        "dtype": "float",
        "shape": [2]
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
        "name": "constantofshape",
        "attributes": {
            "value": {
                "type": "float",
                "data": 1.0
            },
            "dtype": {
                "type": "datatype",
                "data": "float"
            }
        }
      },
      "input": ["a+b"],
      "output": ["constantofshape(a+b)"]
    },
    {
      "info": {
        "type": "op",
        "name": "add"
      },
      "input": ["constantofshape(a+b)", "const1"],
      "output": ["constantofshape(a+b)+const1"]
    }
  ],
  "inputs": ["a", "b"],
  "outputs": ["constantofshape(a+b)+const1"],
  "tensors": [
      {
        "name": "const1",
        "content": {
            "dtype": "float",
            "data": [0.0]
        }
    }
  ]
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        ("a".to_string(), vec![2i64, 3i64].try_into().unwrap()),
        ("b".to_string(), vec![1i64, 1i64].try_into().unwrap()),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("constantofshape(a+b)+const1")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, vec![0.0; 3 * 4]);
}
