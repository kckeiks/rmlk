use rmlk_schema::DataType;
use serde::{Deserialize, Serialize};

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
    Op { name: String },
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
