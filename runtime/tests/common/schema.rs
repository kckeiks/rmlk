use rmlk_schema::{Attribute, AttributeType, DataType, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize)]
pub struct GraphDef {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub nodes: Vec<NodeDef>,
    pub tensors: Vec<TensorDef>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NodeDef {
    pub input: Option<Vec<String>>,
    pub output: Option<Vec<String>>,
    pub info: NodeTypeInfo,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum NodeTypeInfo {
    Op {
        name: String,
        attributes: Option<HashMap<String, AttributeValue>>,
    },
    Value(ValueDef),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ValueDef {
    pub name: String,
    pub shape: Option<Vec<usize>>,
    #[serde(deserialize_with = "deserialize_lowercase")]
    pub dtype: Option<DataType>,
    pub constant: Option<bool>,
}

fn deserialize_lowercase<'de, D>(deserializer: D) -> Result<Option<DataType>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let variant: &str = Deserialize::deserialize(deserializer)?;
    match variant {
        "float" => Ok(Some(DataType::Float)),
        "half" => Ok(Some(DataType::Float16)),
        "int" => Ok(Some(DataType::Int32)),
        _ => Err(serde::de::Error::unknown_variant(
            variant,
            &["float", "half", "int"],
        )),
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TensorDef {
    pub name: String,
    pub content: Data,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "dtype", content = "data")]
#[serde(rename_all = "lowercase")]
pub enum Data {
    Float(Vec<f32>),
    Double(Vec<f64>),
}

impl Data {
    pub fn float(self) -> Vec<f32> {
        let Data::Float(values) = self else {
            panic!("data was not a float");
        };
        values
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", content = "data")]
#[serde(rename_all = "lowercase")]
pub enum AttributeValue {
    Float(f32),
    Int(i32),
    String(Vec<u8>),
    Tensor(Tensor),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    Ints(Vec<i32>),
    Strings(Vec<Vec<u8>>),
    Tensors(Vec<Tensor>),
}

pub fn parse_attributes(attributes: HashMap<String, AttributeValue>) -> Vec<Attribute> {
    let mut res = Vec::new();
    for (name, attribute_value) in attributes {
        let attr_ty = match attribute_value {
            AttributeValue::Float(v) => AttributeType::Float(v),
            AttributeValue::Int(v) => AttributeType::Int(v),
            AttributeValue::String(v) => AttributeType::String(v),
            AttributeValue::Tensor(v) => AttributeType::Tensor(v),
            AttributeValue::Floats(v) => AttributeType::Floats(v),
            AttributeValue::Doubles(v) => AttributeType::Doubles(v),
            AttributeValue::Ints(v) => AttributeType::Ints(v),
            AttributeValue::Strings(v) => AttributeType::Strings(v),
            AttributeValue::Tensors(v) => AttributeType::Tensors(v),
        };
        res.push(Attribute {
            name,
            ref_attr_name: None,
            ty: attr_ty,
            doc_string: None,
        });
    }

    res
}
