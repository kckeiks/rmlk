/*
     "dtype": "<f4",
     "shape": [
       1,
       24,
       27,
       128
     ],
     "stride": [
       82944,
       3456,
       128,
       1
     ]
   },
   {
     "data": "QAAAAAAAAAA=",
     "dtype": "<i8",
     "shape": [
       1
     ],
     "stride": [
       1
     ]
   },
   {
     "data": "/////////38=",
     "dtype": "<i8",
     "shape": [
       1
     ],
     "stride": [
       1
     ]
   },
   {
     "data": "AwAAAAAAAAA=",
     "dtype": "<i8",
     "shape": [
       1
     ],
     "stride": [
       1
     ]
   },
   {
     "data": "AQAAAAAAAAA=",
     "dtype": "<i8",
     "shape": [
       1
     ],
     "stride": [
       1
     ]
   }
*/

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
        "shape": [1, 24, 27, 128]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "starts",
        "dtype": "int64",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "ends",
        "dtype": "int64",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "axes",
        "dtype": "int64",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "steps",
        "dtype": "int64",
        "shape": [1]
      }
    },
    {
      "info": {
        "type": "value",
        "name": "slice(data, starts, ends, axes, steps)",
        "dtype": "float",
        "shape": [1, 24, 27, 64]
      }
    },
    {
      "info": {
        "type": "op",
        "name": "slice"
      },
      "input": ["data", "starts", "ends", "axes", "steps"],
      "output": ["slice(data, starts, ends, axes, steps)"]
    }
  ],
  "inputs": ["data", "starts", "ends", "axes", "steps"],
  "outputs": ["slice(data, starts, ends, axes, steps)"],
  "tensors": []
}
"#;

#[test]
fn test_run() {
    let data_len = 24 * 27 * 128;
    let data_flat: Vec<f32> = (0..data_len).map(|x| x as f32).collect();

    let mut slice_flat: Vec<f32> = Vec::with_capacity(41_472);

    let c_in = 128;
    let inner_len = c_in;

    for pos in 0..(24 * 27) {
        let base = pos * inner_len;
        slice_flat.extend_from_slice(&data_flat[base + 64..base + 128]);
    }

    let input: HashMap<String, Value> = [
        ("data".to_string(), data_flat.clone().into()),
        ("starts".to_string(), vec![64i64].try_into().unwrap()),
        (
            "ends".to_string(),
            vec![9223372036854775807_i64].try_into().unwrap(),
        ),
        ("axes".to_string(), vec![3i64].try_into().unwrap()),
        ("steps".to_string(), vec![1i64].try_into().unwrap()),
    ]
    .into();
    let mut instance = common::build(GRAPH_DEFINITION).build().unwrap();
    let mut output = instance.run(input).unwrap();
    let data: Vec<f32> = output
        .remove("slice(data, starts, ends, axes, steps)")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(data, slice_flat);
}
