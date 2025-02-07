use crate::common;
use rmlk_runtime::Value;
use std::collections::HashMap;

const GRAPH_DEFINITION: &str = r#"
{
  "nodes": [
    {
      "info": {
        "type": "value",
        "name": "data",
        "dtype": "float",
        "shape": [2, 3]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "indices",
        "dtype": "float",
        "shape": [2, 2]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "gather(data, indices)",
        "dtype": "float",
        "shape": [2, 2, 3]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "gather",
        "attributes": {
            "axis": {
                "type": "int",
                "data": 0
            }
        }
      },
      "input": ["data", "indices"],
      "output": ["gather(data, indices)"]
    }
  ],
  "inputs": ["data", "indices"],
  "outputs": ["gather(data, indices)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "data".to_string(),
            vec![
                1.0, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
            ]
            .try_into()
            .unwrap(),
        ),
        (
            "indices".to_string(),
            vec![0i32, 1i32, 1i32, 0i32].try_into().unwrap(),
        ),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("gather(data, indices)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(
        data,
        vec![
            1.0, 2.0, 3.0, // for indices[0,0]=0
            4.0, 5.0, 6.0, // for indices[0,1]=1
            4.0, 5.0, 6.0, // for indices[1,0]=1
            1.0, 2.0, 3.0, // for indices[1,1]=0
        ]
    );
}

#[test]
fn test_run_negative_indices() {
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let input: HashMap<String, Value> = [
        (
            "data".to_string(),
            vec![
                1.0, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
            ]
            .try_into()
            .unwrap(),
        ),
        (
            "indices".to_string(),
            vec![0i32, -1i32, 1i32, -2i32].try_into().unwrap(),
        ),
    ]
    .into();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("gather(data, indices)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(
        data,
        vec![
            1.0, 2.0, 3.0, // for indices[0,0]=0
            4.0, 5.0, 6.0, // for indices[0,1]=1
            4.0, 5.0, 6.0, // for indices[1,0]=1
            1.0, 2.0, 3.0, // for indices[1,1]=0
        ]
    );
}
