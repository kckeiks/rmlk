// Automatically generated rust module for 'onnx.proto' file

#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]
#![allow(unknown_lints)]
#![allow(clippy::all)]
#![cfg_attr(rustfmt, rustfmt_skip)]


use std::borrow::Cow;
use quick_protobuf::{MessageInfo, MessageRead, MessageWrite, BytesReader, Writer, WriterBackend, Result};
use quick_protobuf::sizeofs::*;
use super::*;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Version {
    _START_VERSION = 0,
    IR_VERSION_2017_10_10 = 1,
    IR_VERSION_2017_10_30 = 2,
    IR_VERSION_2017_11_3 = 3,
    IR_VERSION_2019_1_22 = 4,
    IR_VERSION_2019_3_18 = 5,
    IR_VERSION_2019_9_19 = 6,
    IR_VERSION_2020_5_8 = 7,
    IR_VERSION_2021_7_30 = 8,
    IR_VERSION_2023_5_5 = 9,
    IR_VERSION = 10,
}

impl Default for Version {
    fn default() -> Self {
        Version::_START_VERSION
    }
}

impl From<i32> for Version {
    fn from(i: i32) -> Self {
        match i {
            0 => Version::_START_VERSION,
            1 => Version::IR_VERSION_2017_10_10,
            2 => Version::IR_VERSION_2017_10_30,
            3 => Version::IR_VERSION_2017_11_3,
            4 => Version::IR_VERSION_2019_1_22,
            5 => Version::IR_VERSION_2019_3_18,
            6 => Version::IR_VERSION_2019_9_19,
            7 => Version::IR_VERSION_2020_5_8,
            8 => Version::IR_VERSION_2021_7_30,
            9 => Version::IR_VERSION_2023_5_5,
            10 => Version::IR_VERSION,
            _ => Self::default(),
        }
    }
}

impl<'a> From<&'a str> for Version {
    fn from(s: &'a str) -> Self {
        match s {
            "_START_VERSION" => Version::_START_VERSION,
            "IR_VERSION_2017_10_10" => Version::IR_VERSION_2017_10_10,
            "IR_VERSION_2017_10_30" => Version::IR_VERSION_2017_10_30,
            "IR_VERSION_2017_11_3" => Version::IR_VERSION_2017_11_3,
            "IR_VERSION_2019_1_22" => Version::IR_VERSION_2019_1_22,
            "IR_VERSION_2019_3_18" => Version::IR_VERSION_2019_3_18,
            "IR_VERSION_2019_9_19" => Version::IR_VERSION_2019_9_19,
            "IR_VERSION_2020_5_8" => Version::IR_VERSION_2020_5_8,
            "IR_VERSION_2021_7_30" => Version::IR_VERSION_2021_7_30,
            "IR_VERSION_2023_5_5" => Version::IR_VERSION_2023_5_5,
            "IR_VERSION" => Version::IR_VERSION,
            _ => Self::default(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum OperatorStatus {
    EXPERIMENTAL = 0,
    STABLE = 1,
}

impl Default for OperatorStatus {
    fn default() -> Self {
        OperatorStatus::EXPERIMENTAL
    }
}

impl From<i32> for OperatorStatus {
    fn from(i: i32) -> Self {
        match i {
            0 => OperatorStatus::EXPERIMENTAL,
            1 => OperatorStatus::STABLE,
            _ => Self::default(),
        }
    }
}

impl<'a> From<&'a str> for OperatorStatus {
    fn from(s: &'a str) -> Self {
        match s {
            "EXPERIMENTAL" => OperatorStatus::EXPERIMENTAL,
            "STABLE" => OperatorStatus::STABLE,
            _ => Self::default(),
        }
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct AttributeProto<'a> {
    pub name: Option<Cow<'a, str>>,
    pub ref_attr_name: Option<Cow<'a, str>>,
    pub doc_string: Option<Cow<'a, str>>,
    pub type_pb: Option<mod_AttributeProto::AttributeType>,
    pub f: Option<f32>,
    pub i: Option<i64>,
    pub s: Option<Cow<'a, [u8]>>,
    pub t: Option<TensorProto<'a>>,
    pub g: Option<GraphProto<'a>>,
    pub sparse_tensor: Option<SparseTensorProto<'a>>,
    pub tp: Option<TypeProto<'a>>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<Cow<'a, [u8]>>,
    pub tensors: Vec<TensorProto<'a>>,
    pub graphs: Vec<GraphProto<'a>>,
    pub sparse_tensors: Vec<SparseTensorProto<'a>>,
    pub type_protos: Vec<TypeProto<'a>>,
}

impl<'a> MessageRead<'a> for AttributeProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.name = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(170) => msg.ref_attr_name = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(106) => msg.doc_string = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(160) => msg.type_pb = Some(r.read_enum(bytes)?),
                Ok(21) => msg.f = Some(r.read_float(bytes)?),
                Ok(24) => msg.i = Some(r.read_int64(bytes)?),
                Ok(34) => msg.s = Some(r.read_bytes(bytes).map(Cow::Borrowed)?),
                Ok(42) => msg.t = Some(r.read_message::<TensorProto>(bytes)?),
                Ok(50) => msg.g = Some(r.read_message::<GraphProto>(bytes)?),
                Ok(178) => msg.sparse_tensor = Some(r.read_message::<SparseTensorProto>(bytes)?),
                Ok(114) => msg.tp = Some(r.read_message::<TypeProto>(bytes)?),
                Ok(61) => msg.floats.push(r.read_float(bytes)?),
                Ok(64) => msg.ints.push(r.read_int64(bytes)?),
                Ok(74) => msg.strings.push(r.read_bytes(bytes).map(Cow::Borrowed)?),
                Ok(82) => msg.tensors.push(r.read_message::<TensorProto>(bytes)?),
                Ok(90) => msg.graphs.push(r.read_message::<GraphProto>(bytes)?),
                Ok(186) => msg.sparse_tensors.push(r.read_message::<SparseTensorProto>(bytes)?),
                Ok(122) => msg.type_protos.push(r.read_message::<TypeProto>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for AttributeProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.name.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.ref_attr_name.as_ref().map_or(0, |m| 2 + sizeof_len((m).len()))
        + self.doc_string.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.type_pb.as_ref().map_or(0, |m| 2 + sizeof_varint(*(m) as u64))
        + self.f.as_ref().map_or(0, |_| 1 + 4)
        + self.i.as_ref().map_or(0, |m| 1 + sizeof_varint(*(m) as u64))
        + self.s.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.t.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
        + self.g.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
        + self.sparse_tensor.as_ref().map_or(0, |m| 2 + sizeof_len((m).get_size()))
        + self.tp.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
        + (1 + 4) * self.floats.len()
        + self.ints.iter().map(|s| 1 + sizeof_varint(*(s) as u64)).sum::<usize>()
        + self.strings.iter().map(|s| 1 + sizeof_len((s).len())).sum::<usize>()
        + self.tensors.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.graphs.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.sparse_tensors.iter().map(|s| 2 + sizeof_len((s).get_size())).sum::<usize>()
        + self.type_protos.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.name { w.write_with_tag(10, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.ref_attr_name { w.write_with_tag(170, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.doc_string { w.write_with_tag(106, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.type_pb { w.write_with_tag(160, |w| w.write_enum(*s as i32))?; }
        if let Some(ref s) = self.f { w.write_with_tag(21, |w| w.write_float(*s))?; }
        if let Some(ref s) = self.i { w.write_with_tag(24, |w| w.write_int64(*s))?; }
        if let Some(ref s) = self.s { w.write_with_tag(34, |w| w.write_bytes(&**s))?; }
        if let Some(ref s) = self.t { w.write_with_tag(42, |w| w.write_message(s))?; }
        if let Some(ref s) = self.g { w.write_with_tag(50, |w| w.write_message(s))?; }
        if let Some(ref s) = self.sparse_tensor { w.write_with_tag(178, |w| w.write_message(s))?; }
        if let Some(ref s) = self.tp { w.write_with_tag(114, |w| w.write_message(s))?; }
        for s in &self.floats { w.write_with_tag(61, |w| w.write_float(*s))?; }
        for s in &self.ints { w.write_with_tag(64, |w| w.write_int64(*s))?; }
        for s in &self.strings { w.write_with_tag(74, |w| w.write_bytes(&**s))?; }
        for s in &self.tensors { w.write_with_tag(82, |w| w.write_message(s))?; }
        for s in &self.graphs { w.write_with_tag(90, |w| w.write_message(s))?; }
        for s in &self.sparse_tensors { w.write_with_tag(186, |w| w.write_message(s))?; }
        for s in &self.type_protos { w.write_with_tag(122, |w| w.write_message(s))?; }
        Ok(())
    }
}

pub mod mod_AttributeProto {


#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AttributeType {
    UNDEFINED = 0,
    FLOAT = 1,
    INT = 2,
    STRING = 3,
    TENSOR = 4,
    GRAPH = 5,
    SPARSE_TENSOR = 11,
    TYPE_PROTO = 13,
    FLOATS = 6,
    INTS = 7,
    STRINGS = 8,
    TENSORS = 9,
    GRAPHS = 10,
    SPARSE_TENSORS = 12,
    TYPE_PROTOS = 14,
}

impl Default for AttributeType {
    fn default() -> Self {
        AttributeType::UNDEFINED
    }
}

impl From<i32> for AttributeType {
    fn from(i: i32) -> Self {
        match i {
            0 => AttributeType::UNDEFINED,
            1 => AttributeType::FLOAT,
            2 => AttributeType::INT,
            3 => AttributeType::STRING,
            4 => AttributeType::TENSOR,
            5 => AttributeType::GRAPH,
            11 => AttributeType::SPARSE_TENSOR,
            13 => AttributeType::TYPE_PROTO,
            6 => AttributeType::FLOATS,
            7 => AttributeType::INTS,
            8 => AttributeType::STRINGS,
            9 => AttributeType::TENSORS,
            10 => AttributeType::GRAPHS,
            12 => AttributeType::SPARSE_TENSORS,
            14 => AttributeType::TYPE_PROTOS,
            _ => Self::default(),
        }
    }
}

impl<'a> From<&'a str> for AttributeType {
    fn from(s: &'a str) -> Self {
        match s {
            "UNDEFINED" => AttributeType::UNDEFINED,
            "FLOAT" => AttributeType::FLOAT,
            "INT" => AttributeType::INT,
            "STRING" => AttributeType::STRING,
            "TENSOR" => AttributeType::TENSOR,
            "GRAPH" => AttributeType::GRAPH,
            "SPARSE_TENSOR" => AttributeType::SPARSE_TENSOR,
            "TYPE_PROTO" => AttributeType::TYPE_PROTO,
            "FLOATS" => AttributeType::FLOATS,
            "INTS" => AttributeType::INTS,
            "STRINGS" => AttributeType::STRINGS,
            "TENSORS" => AttributeType::TENSORS,
            "GRAPHS" => AttributeType::GRAPHS,
            "SPARSE_TENSORS" => AttributeType::SPARSE_TENSORS,
            "TYPE_PROTOS" => AttributeType::TYPE_PROTOS,
            _ => Self::default(),
        }
    }
}

}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct ValueInfoProto<'a> {
    pub name: Option<Cow<'a, str>>,
    pub type_pb: Option<TypeProto<'a>>,
    pub doc_string: Option<Cow<'a, str>>,
    pub metadata_props: Vec<StringStringEntryProto<'a>>,
}

impl<'a> MessageRead<'a> for ValueInfoProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.name = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(18) => msg.type_pb = Some(r.read_message::<TypeProto>(bytes)?),
                Ok(26) => msg.doc_string = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(34) => msg.metadata_props.push(r.read_message::<StringStringEntryProto>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for ValueInfoProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.name.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.type_pb.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
        + self.doc_string.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.metadata_props.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.name { w.write_with_tag(10, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.type_pb { w.write_with_tag(18, |w| w.write_message(s))?; }
        if let Some(ref s) = self.doc_string { w.write_with_tag(26, |w| w.write_string(&**s))?; }
        for s in &self.metadata_props { w.write_with_tag(34, |w| w.write_message(s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct NodeProto<'a> {
    pub input: Vec<Cow<'a, str>>,
    pub output: Vec<Cow<'a, str>>,
    pub name: Option<Cow<'a, str>>,
    pub op_type: Option<Cow<'a, str>>,
    pub domain: Option<Cow<'a, str>>,
    pub overload: Option<Cow<'a, str>>,
    pub attribute: Vec<AttributeProto<'a>>,
    pub doc_string: Option<Cow<'a, str>>,
    pub metadata_props: Vec<StringStringEntryProto<'a>>,
}

impl<'a> MessageRead<'a> for NodeProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.input.push(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(18) => msg.output.push(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(26) => msg.name = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(34) => msg.op_type = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(58) => msg.domain = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(66) => msg.overload = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(42) => msg.attribute.push(r.read_message::<AttributeProto>(bytes)?),
                Ok(50) => msg.doc_string = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(74) => msg.metadata_props.push(r.read_message::<StringStringEntryProto>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for NodeProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.input.iter().map(|s| 1 + sizeof_len((s).len())).sum::<usize>()
        + self.output.iter().map(|s| 1 + sizeof_len((s).len())).sum::<usize>()
        + self.name.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.op_type.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.domain.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.overload.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.attribute.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.doc_string.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.metadata_props.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        for s in &self.input { w.write_with_tag(10, |w| w.write_string(&**s))?; }
        for s in &self.output { w.write_with_tag(18, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.name { w.write_with_tag(26, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.op_type { w.write_with_tag(34, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.domain { w.write_with_tag(58, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.overload { w.write_with_tag(66, |w| w.write_string(&**s))?; }
        for s in &self.attribute { w.write_with_tag(42, |w| w.write_message(s))?; }
        if let Some(ref s) = self.doc_string { w.write_with_tag(50, |w| w.write_string(&**s))?; }
        for s in &self.metadata_props { w.write_with_tag(74, |w| w.write_message(s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct TrainingInfoProto<'a> {
    pub initialization: Option<GraphProto<'a>>,
    pub algorithm: Option<GraphProto<'a>>,
    pub initialization_binding: Vec<StringStringEntryProto<'a>>,
    pub update_binding: Vec<StringStringEntryProto<'a>>,
}

impl<'a> MessageRead<'a> for TrainingInfoProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.initialization = Some(r.read_message::<GraphProto>(bytes)?),
                Ok(18) => msg.algorithm = Some(r.read_message::<GraphProto>(bytes)?),
                Ok(26) => msg.initialization_binding.push(r.read_message::<StringStringEntryProto>(bytes)?),
                Ok(34) => msg.update_binding.push(r.read_message::<StringStringEntryProto>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for TrainingInfoProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.initialization.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
        + self.algorithm.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
        + self.initialization_binding.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.update_binding.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.initialization { w.write_with_tag(10, |w| w.write_message(s))?; }
        if let Some(ref s) = self.algorithm { w.write_with_tag(18, |w| w.write_message(s))?; }
        for s in &self.initialization_binding { w.write_with_tag(26, |w| w.write_message(s))?; }
        for s in &self.update_binding { w.write_with_tag(34, |w| w.write_message(s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct ModelProto<'a> {
    pub ir_version: Option<i64>,
    pub opset_import: Vec<OperatorSetIdProto<'a>>,
    pub producer_name: Option<Cow<'a, str>>,
    pub producer_version: Option<Cow<'a, str>>,
    pub domain: Option<Cow<'a, str>>,
    pub model_version: Option<i64>,
    pub doc_string: Option<Cow<'a, str>>,
    pub graph: Option<GraphProto<'a>>,
    pub metadata_props: Vec<StringStringEntryProto<'a>>,
    pub training_info: Vec<TrainingInfoProto<'a>>,
    pub functions: Vec<FunctionProto<'a>>,
}

impl<'a> MessageRead<'a> for ModelProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(8) => msg.ir_version = Some(r.read_int64(bytes)?),
                Ok(66) => msg.opset_import.push(r.read_message::<OperatorSetIdProto>(bytes)?),
                Ok(18) => msg.producer_name = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(26) => msg.producer_version = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(34) => msg.domain = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(40) => msg.model_version = Some(r.read_int64(bytes)?),
                Ok(50) => msg.doc_string = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(58) => msg.graph = Some(r.read_message::<GraphProto>(bytes)?),
                Ok(114) => msg.metadata_props.push(r.read_message::<StringStringEntryProto>(bytes)?),
                Ok(162) => msg.training_info.push(r.read_message::<TrainingInfoProto>(bytes)?),
                Ok(202) => msg.functions.push(r.read_message::<FunctionProto>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for ModelProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.ir_version.as_ref().map_or(0, |m| 1 + sizeof_varint(*(m) as u64))
        + self.opset_import.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.producer_name.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.producer_version.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.domain.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.model_version.as_ref().map_or(0, |m| 1 + sizeof_varint(*(m) as u64))
        + self.doc_string.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.graph.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
        + self.metadata_props.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.training_info.iter().map(|s| 2 + sizeof_len((s).get_size())).sum::<usize>()
        + self.functions.iter().map(|s| 2 + sizeof_len((s).get_size())).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.ir_version { w.write_with_tag(8, |w| w.write_int64(*s))?; }
        for s in &self.opset_import { w.write_with_tag(66, |w| w.write_message(s))?; }
        if let Some(ref s) = self.producer_name { w.write_with_tag(18, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.producer_version { w.write_with_tag(26, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.domain { w.write_with_tag(34, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.model_version { w.write_with_tag(40, |w| w.write_int64(*s))?; }
        if let Some(ref s) = self.doc_string { w.write_with_tag(50, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.graph { w.write_with_tag(58, |w| w.write_message(s))?; }
        for s in &self.metadata_props { w.write_with_tag(114, |w| w.write_message(s))?; }
        for s in &self.training_info { w.write_with_tag(162, |w| w.write_message(s))?; }
        for s in &self.functions { w.write_with_tag(202, |w| w.write_message(s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct StringStringEntryProto<'a> {
    pub key: Option<Cow<'a, str>>,
    pub value: Option<Cow<'a, str>>,
}

impl<'a> MessageRead<'a> for StringStringEntryProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.key = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(18) => msg.value = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for StringStringEntryProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.key.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.value.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.key { w.write_with_tag(10, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.value { w.write_with_tag(18, |w| w.write_string(&**s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct TensorAnnotation<'a> {
    pub tensor_name: Option<Cow<'a, str>>,
    pub quant_parameter_tensor_names: Vec<StringStringEntryProto<'a>>,
}

impl<'a> MessageRead<'a> for TensorAnnotation<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.tensor_name = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(18) => msg.quant_parameter_tensor_names.push(r.read_message::<StringStringEntryProto>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for TensorAnnotation<'a> {
    fn get_size(&self) -> usize {
        0
        + self.tensor_name.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.quant_parameter_tensor_names.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.tensor_name { w.write_with_tag(10, |w| w.write_string(&**s))?; }
        for s in &self.quant_parameter_tensor_names { w.write_with_tag(18, |w| w.write_message(s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct GraphProto<'a> {
    pub node: Vec<NodeProto<'a>>,
    pub name: Option<Cow<'a, str>>,
    pub initializer: Vec<TensorProto<'a>>,
    pub sparse_initializer: Vec<SparseTensorProto<'a>>,
    pub doc_string: Option<Cow<'a, str>>,
    pub input: Vec<ValueInfoProto<'a>>,
    pub output: Vec<ValueInfoProto<'a>>,
    pub value_info: Vec<ValueInfoProto<'a>>,
    pub quantization_annotation: Vec<TensorAnnotation<'a>>,
    pub metadata_props: Vec<StringStringEntryProto<'a>>,
}

impl<'a> MessageRead<'a> for GraphProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.node.push(r.read_message::<NodeProto>(bytes)?),
                Ok(18) => msg.name = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(42) => msg.initializer.push(r.read_message::<TensorProto>(bytes)?),
                Ok(122) => msg.sparse_initializer.push(r.read_message::<SparseTensorProto>(bytes)?),
                Ok(82) => msg.doc_string = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(90) => msg.input.push(r.read_message::<ValueInfoProto>(bytes)?),
                Ok(98) => msg.output.push(r.read_message::<ValueInfoProto>(bytes)?),
                Ok(106) => msg.value_info.push(r.read_message::<ValueInfoProto>(bytes)?),
                Ok(114) => msg.quantization_annotation.push(r.read_message::<TensorAnnotation>(bytes)?),
                Ok(130) => msg.metadata_props.push(r.read_message::<StringStringEntryProto>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for GraphProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.node.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.name.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.initializer.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.sparse_initializer.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.doc_string.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.input.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.output.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.value_info.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.quantization_annotation.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.metadata_props.iter().map(|s| 2 + sizeof_len((s).get_size())).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        for s in &self.node { w.write_with_tag(10, |w| w.write_message(s))?; }
        if let Some(ref s) = self.name { w.write_with_tag(18, |w| w.write_string(&**s))?; }
        for s in &self.initializer { w.write_with_tag(42, |w| w.write_message(s))?; }
        for s in &self.sparse_initializer { w.write_with_tag(122, |w| w.write_message(s))?; }
        if let Some(ref s) = self.doc_string { w.write_with_tag(82, |w| w.write_string(&**s))?; }
        for s in &self.input { w.write_with_tag(90, |w| w.write_message(s))?; }
        for s in &self.output { w.write_with_tag(98, |w| w.write_message(s))?; }
        for s in &self.value_info { w.write_with_tag(106, |w| w.write_message(s))?; }
        for s in &self.quantization_annotation { w.write_with_tag(114, |w| w.write_message(s))?; }
        for s in &self.metadata_props { w.write_with_tag(130, |w| w.write_message(s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct TensorProto<'a> {
    pub dims: Vec<i64>,
    pub data_type: Option<i32>,
    pub segment: Option<mod_TensorProto::Segment>,
    pub float_data: Cow<'a, [f32]>,
    pub int32_data: Vec<i32>,
    pub string_data: Vec<Cow<'a, [u8]>>,
    pub int64_data: Vec<i64>,
    pub name: Option<Cow<'a, str>>,
    pub doc_string: Option<Cow<'a, str>>,
    pub raw_data: Option<Cow<'a, [u8]>>,
    pub external_data: Vec<StringStringEntryProto<'a>>,
    pub data_location: Option<mod_TensorProto::DataLocation>,
    pub double_data: Cow<'a, [f64]>,
    pub uint64_data: Vec<u64>,
    pub metadata_props: Vec<StringStringEntryProto<'a>>,
}

impl<'a> MessageRead<'a> for TensorProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(8) => msg.dims.push(r.read_int64(bytes)?),
                Ok(16) => msg.data_type = Some(r.read_int32(bytes)?),
                Ok(26) => msg.segment = Some(r.read_message::<mod_TensorProto::Segment>(bytes)?),
                Ok(34) => msg.float_data = r.read_packed_fixed(bytes)?.into(),
                Ok(42) => msg.int32_data = r.read_packed(bytes, |r, bytes| Ok(r.read_int32(bytes)?))?,
                Ok(50) => msg.string_data.push(r.read_bytes(bytes).map(Cow::Borrowed)?),
                Ok(58) => msg.int64_data = r.read_packed(bytes, |r, bytes| Ok(r.read_int64(bytes)?))?,
                Ok(66) => msg.name = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(98) => msg.doc_string = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(74) => msg.raw_data = Some(r.read_bytes(bytes).map(Cow::Borrowed)?),
                Ok(106) => msg.external_data.push(r.read_message::<StringStringEntryProto>(bytes)?),
                Ok(112) => msg.data_location = Some(r.read_enum(bytes)?),
                Ok(82) => msg.double_data = r.read_packed_fixed(bytes)?.into(),
                Ok(90) => msg.uint64_data = r.read_packed(bytes, |r, bytes| Ok(r.read_uint64(bytes)?))?,
                Ok(130) => msg.metadata_props.push(r.read_message::<StringStringEntryProto>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for TensorProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.dims.iter().map(|s| 1 + sizeof_varint(*(s) as u64)).sum::<usize>()
        + self.data_type.as_ref().map_or(0, |m| 1 + sizeof_varint(*(m) as u64))
        + self.segment.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
        + if self.float_data.is_empty() { 0 } else { 1 + sizeof_len(self.float_data.len() * 4) }
        + if self.int32_data.is_empty() { 0 } else { 1 + sizeof_len(self.int32_data.iter().map(|s| sizeof_varint(*(s) as u64)).sum::<usize>()) }
        + self.string_data.iter().map(|s| 1 + sizeof_len((s).len())).sum::<usize>()
        + if self.int64_data.is_empty() { 0 } else { 1 + sizeof_len(self.int64_data.iter().map(|s| sizeof_varint(*(s) as u64)).sum::<usize>()) }
        + self.name.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.doc_string.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.raw_data.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.external_data.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.data_location.as_ref().map_or(0, |m| 1 + sizeof_varint(*(m) as u64))
        + if self.double_data.is_empty() { 0 } else { 1 + sizeof_len(self.double_data.len() * 8) }
        + if self.uint64_data.is_empty() { 0 } else { 1 + sizeof_len(self.uint64_data.iter().map(|s| sizeof_varint(*(s) as u64)).sum::<usize>()) }
        + self.metadata_props.iter().map(|s| 2 + sizeof_len((s).get_size())).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        for s in &self.dims { w.write_with_tag(8, |w| w.write_int64(*s))?; }
        if let Some(ref s) = self.data_type { w.write_with_tag(16, |w| w.write_int32(*s))?; }
        if let Some(ref s) = self.segment { w.write_with_tag(26, |w| w.write_message(s))?; }
        w.write_packed_fixed_with_tag(34, &self.float_data)?;
        w.write_packed_with_tag(42, &self.int32_data, |w, m| w.write_int32(*m), &|m| sizeof_varint(*(m) as u64))?;
        for s in &self.string_data { w.write_with_tag(50, |w| w.write_bytes(&**s))?; }
        w.write_packed_with_tag(58, &self.int64_data, |w, m| w.write_int64(*m), &|m| sizeof_varint(*(m) as u64))?;
        if let Some(ref s) = self.name { w.write_with_tag(66, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.doc_string { w.write_with_tag(98, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.raw_data { w.write_with_tag(74, |w| w.write_bytes(&**s))?; }
        for s in &self.external_data { w.write_with_tag(106, |w| w.write_message(s))?; }
        if let Some(ref s) = self.data_location { w.write_with_tag(112, |w| w.write_enum(*s as i32))?; }
        w.write_packed_fixed_with_tag(82, &self.double_data)?;
        w.write_packed_with_tag(90, &self.uint64_data, |w, m| w.write_uint64(*m), &|m| sizeof_varint(*(m) as u64))?;
        for s in &self.metadata_props { w.write_with_tag(130, |w| w.write_message(s))?; }
        Ok(())
    }
}

pub mod mod_TensorProto {

use super::*;

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Segment {
    pub begin: Option<i64>,
    pub end: Option<i64>,
}

impl<'a> MessageRead<'a> for Segment {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(8) => msg.begin = Some(r.read_int64(bytes)?),
                Ok(16) => msg.end = Some(r.read_int64(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl MessageWrite for Segment {
    fn get_size(&self) -> usize {
        0
        + self.begin.as_ref().map_or(0, |m| 1 + sizeof_varint(*(m) as u64))
        + self.end.as_ref().map_or(0, |m| 1 + sizeof_varint(*(m) as u64))
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.begin { w.write_with_tag(8, |w| w.write_int64(*s))?; }
        if let Some(ref s) = self.end { w.write_with_tag(16, |w| w.write_int64(*s))?; }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DataType {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
    FLOAT8E4M3FN = 17,
    FLOAT8E4M3FNUZ = 18,
    FLOAT8E5M2 = 19,
    FLOAT8E5M2FNUZ = 20,
    UINT4 = 21,
    INT4 = 22,
}

impl Default for DataType {
    fn default() -> Self {
        DataType::UNDEFINED
    }
}

impl From<i32> for DataType {
    fn from(i: i32) -> Self {
        match i {
            0 => DataType::UNDEFINED,
            1 => DataType::FLOAT,
            2 => DataType::UINT8,
            3 => DataType::INT8,
            4 => DataType::UINT16,
            5 => DataType::INT16,
            6 => DataType::INT32,
            7 => DataType::INT64,
            8 => DataType::STRING,
            9 => DataType::BOOL,
            10 => DataType::FLOAT16,
            11 => DataType::DOUBLE,
            12 => DataType::UINT32,
            13 => DataType::UINT64,
            14 => DataType::COMPLEX64,
            15 => DataType::COMPLEX128,
            16 => DataType::BFLOAT16,
            17 => DataType::FLOAT8E4M3FN,
            18 => DataType::FLOAT8E4M3FNUZ,
            19 => DataType::FLOAT8E5M2,
            20 => DataType::FLOAT8E5M2FNUZ,
            21 => DataType::UINT4,
            22 => DataType::INT4,
            _ => Self::default(),
        }
    }
}

impl<'a> From<&'a str> for DataType {
    fn from(s: &'a str) -> Self {
        match s {
            "UNDEFINED" => DataType::UNDEFINED,
            "FLOAT" => DataType::FLOAT,
            "UINT8" => DataType::UINT8,
            "INT8" => DataType::INT8,
            "UINT16" => DataType::UINT16,
            "INT16" => DataType::INT16,
            "INT32" => DataType::INT32,
            "INT64" => DataType::INT64,
            "STRING" => DataType::STRING,
            "BOOL" => DataType::BOOL,
            "FLOAT16" => DataType::FLOAT16,
            "DOUBLE" => DataType::DOUBLE,
            "UINT32" => DataType::UINT32,
            "UINT64" => DataType::UINT64,
            "COMPLEX64" => DataType::COMPLEX64,
            "COMPLEX128" => DataType::COMPLEX128,
            "BFLOAT16" => DataType::BFLOAT16,
            "FLOAT8E4M3FN" => DataType::FLOAT8E4M3FN,
            "FLOAT8E4M3FNUZ" => DataType::FLOAT8E4M3FNUZ,
            "FLOAT8E5M2" => DataType::FLOAT8E5M2,
            "FLOAT8E5M2FNUZ" => DataType::FLOAT8E5M2FNUZ,
            "UINT4" => DataType::UINT4,
            "INT4" => DataType::INT4,
            _ => Self::default(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DataLocation {
    DEFAULT = 0,
    EXTERNAL = 1,
}

impl Default for DataLocation {
    fn default() -> Self {
        DataLocation::DEFAULT
    }
}

impl From<i32> for DataLocation {
    fn from(i: i32) -> Self {
        match i {
            0 => DataLocation::DEFAULT,
            1 => DataLocation::EXTERNAL,
            _ => Self::default(),
        }
    }
}

impl<'a> From<&'a str> for DataLocation {
    fn from(s: &'a str) -> Self {
        match s {
            "DEFAULT" => DataLocation::DEFAULT,
            "EXTERNAL" => DataLocation::EXTERNAL,
            _ => Self::default(),
        }
    }
}

}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct SparseTensorProto<'a> {
    pub values: Option<TensorProto<'a>>,
    pub indices: Option<TensorProto<'a>>,
    pub dims: Vec<i64>,
}

impl<'a> MessageRead<'a> for SparseTensorProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.values = Some(r.read_message::<TensorProto>(bytes)?),
                Ok(18) => msg.indices = Some(r.read_message::<TensorProto>(bytes)?),
                Ok(24) => msg.dims.push(r.read_int64(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for SparseTensorProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.values.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
        + self.indices.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
        + self.dims.iter().map(|s| 1 + sizeof_varint(*(s) as u64)).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.values { w.write_with_tag(10, |w| w.write_message(s))?; }
        if let Some(ref s) = self.indices { w.write_with_tag(18, |w| w.write_message(s))?; }
        for s in &self.dims { w.write_with_tag(24, |w| w.write_int64(*s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct TensorShapeProto<'a> {
    pub dim: Vec<mod_TensorShapeProto::Dimension<'a>>,
}

impl<'a> MessageRead<'a> for TensorShapeProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.dim.push(r.read_message::<mod_TensorShapeProto::Dimension>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for TensorShapeProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.dim.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        for s in &self.dim { w.write_with_tag(10, |w| w.write_message(s))?; }
        Ok(())
    }
}

pub mod mod_TensorShapeProto {

use std::borrow::Cow;
use super::*;

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Dimension<'a> {
    pub denotation: Option<Cow<'a, str>>,
    pub value: mod_TensorShapeProto::mod_Dimension::OneOfvalue<'a>,
}

impl<'a> MessageRead<'a> for Dimension<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(26) => msg.denotation = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(8) => msg.value = mod_TensorShapeProto::mod_Dimension::OneOfvalue::dim_value(r.read_int64(bytes)?),
                Ok(18) => msg.value = mod_TensorShapeProto::mod_Dimension::OneOfvalue::dim_param(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for Dimension<'a> {
    fn get_size(&self) -> usize {
        0
        + self.denotation.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + match self.value {
            mod_TensorShapeProto::mod_Dimension::OneOfvalue::dim_value(ref m) => 1 + sizeof_varint(*(m) as u64),
            mod_TensorShapeProto::mod_Dimension::OneOfvalue::dim_param(ref m) => 1 + sizeof_len((m).len()),
            mod_TensorShapeProto::mod_Dimension::OneOfvalue::None => 0,
    }    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.denotation { w.write_with_tag(26, |w| w.write_string(&**s))?; }
        match self.value {            mod_TensorShapeProto::mod_Dimension::OneOfvalue::dim_value(ref m) => { w.write_with_tag(8, |w| w.write_int64(*m))? },
            mod_TensorShapeProto::mod_Dimension::OneOfvalue::dim_param(ref m) => { w.write_with_tag(18, |w| w.write_string(&**m))? },
            mod_TensorShapeProto::mod_Dimension::OneOfvalue::None => {},
    }        Ok(())
    }
}

pub mod mod_Dimension {

use super::*;

#[derive(Debug, PartialEq, Clone)]
pub enum OneOfvalue<'a> {
    dim_value(i64),
    dim_param(Cow<'a, str>),
    None,
}

impl<'a> Default for OneOfvalue<'a> {
    fn default() -> Self {
        OneOfvalue::None
    }
}

}

}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct TypeProto<'a> {
    pub denotation: Option<Cow<'a, str>>,
    pub value: mod_TypeProto::OneOfvalue<'a>,
}

impl<'a> MessageRead<'a> for TypeProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(50) => msg.denotation = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(10) => msg.value = mod_TypeProto::OneOfvalue::tensor_type(r.read_message::<mod_TypeProto::Tensor>(bytes)?),
                Ok(34) => msg.value = mod_TypeProto::OneOfvalue::sequence_type(Box::new(r.read_message::<mod_TypeProto::Sequence>(bytes)?)),
                Ok(42) => msg.value = mod_TypeProto::OneOfvalue::map_type(Box::new(r.read_message::<mod_TypeProto::Map>(bytes)?)),
                Ok(74) => msg.value = mod_TypeProto::OneOfvalue::optional_type(Box::new(r.read_message::<mod_TypeProto::Optional>(bytes)?)),
                Ok(66) => msg.value = mod_TypeProto::OneOfvalue::sparse_tensor_type(r.read_message::<mod_TypeProto::SparseTensor>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for TypeProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.denotation.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + match self.value {
            mod_TypeProto::OneOfvalue::tensor_type(ref m) => 1 + sizeof_len((m).get_size()),
            mod_TypeProto::OneOfvalue::sequence_type(ref m) => 1 + sizeof_len((m).get_size()),
            mod_TypeProto::OneOfvalue::map_type(ref m) => 1 + sizeof_len((m).get_size()),
            mod_TypeProto::OneOfvalue::optional_type(ref m) => 1 + sizeof_len((m).get_size()),
            mod_TypeProto::OneOfvalue::sparse_tensor_type(ref m) => 1 + sizeof_len((m).get_size()),
            mod_TypeProto::OneOfvalue::None => 0,
    }    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.denotation { w.write_with_tag(50, |w| w.write_string(&**s))?; }
        match self.value {            mod_TypeProto::OneOfvalue::tensor_type(ref m) => { w.write_with_tag(10, |w| w.write_message(m))? },
            mod_TypeProto::OneOfvalue::sequence_type(ref m) => { w.write_with_tag(34, |w| w.write_message(&**m))? },
            mod_TypeProto::OneOfvalue::map_type(ref m) => { w.write_with_tag(42, |w| w.write_message(&**m))? },
            mod_TypeProto::OneOfvalue::optional_type(ref m) => { w.write_with_tag(74, |w| w.write_message(&**m))? },
            mod_TypeProto::OneOfvalue::sparse_tensor_type(ref m) => { w.write_with_tag(66, |w| w.write_message(m))? },
            mod_TypeProto::OneOfvalue::None => {},
    }        Ok(())
    }
}

pub mod mod_TypeProto {

use super::*;

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Tensor<'a> {
    pub elem_type: Option<i32>,
    pub shape: Option<TensorShapeProto<'a>>,
}

impl<'a> MessageRead<'a> for Tensor<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(8) => msg.elem_type = Some(r.read_int32(bytes)?),
                Ok(18) => msg.shape = Some(r.read_message::<TensorShapeProto>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for Tensor<'a> {
    fn get_size(&self) -> usize {
        0
        + self.elem_type.as_ref().map_or(0, |m| 1 + sizeof_varint(*(m) as u64))
        + self.shape.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.elem_type { w.write_with_tag(8, |w| w.write_int32(*s))?; }
        if let Some(ref s) = self.shape { w.write_with_tag(18, |w| w.write_message(s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Sequence<'a> {
    pub elem_type: Option<Box<TypeProto<'a>>>,
}

impl<'a> MessageRead<'a> for Sequence<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.elem_type = Some(Box::new(r.read_message::<TypeProto>(bytes)?)),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for Sequence<'a> {
    fn get_size(&self) -> usize {
        0
        + self.elem_type.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.elem_type { w.write_with_tag(10, |w| w.write_message(&**s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Map<'a> {
    pub key_type: Option<i32>,
    pub value_type: Option<Box<TypeProto<'a>>>,
}

impl<'a> MessageRead<'a> for Map<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(8) => msg.key_type = Some(r.read_int32(bytes)?),
                Ok(18) => msg.value_type = Some(Box::new(r.read_message::<TypeProto>(bytes)?)),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for Map<'a> {
    fn get_size(&self) -> usize {
        0
        + self.key_type.as_ref().map_or(0, |m| 1 + sizeof_varint(*(m) as u64))
        + self.value_type.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.key_type { w.write_with_tag(8, |w| w.write_int32(*s))?; }
        if let Some(ref s) = self.value_type { w.write_with_tag(18, |w| w.write_message(&**s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Optional<'a> {
    pub elem_type: Option<Box<TypeProto<'a>>>,
}

impl<'a> MessageRead<'a> for Optional<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.elem_type = Some(Box::new(r.read_message::<TypeProto>(bytes)?)),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for Optional<'a> {
    fn get_size(&self) -> usize {
        0
        + self.elem_type.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.elem_type { w.write_with_tag(10, |w| w.write_message(&**s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct SparseTensor<'a> {
    pub elem_type: Option<i32>,
    pub shape: Option<TensorShapeProto<'a>>,
}

impl<'a> MessageRead<'a> for SparseTensor<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(8) => msg.elem_type = Some(r.read_int32(bytes)?),
                Ok(18) => msg.shape = Some(r.read_message::<TensorShapeProto>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for SparseTensor<'a> {
    fn get_size(&self) -> usize {
        0
        + self.elem_type.as_ref().map_or(0, |m| 1 + sizeof_varint(*(m) as u64))
        + self.shape.as_ref().map_or(0, |m| 1 + sizeof_len((m).get_size()))
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.elem_type { w.write_with_tag(8, |w| w.write_int32(*s))?; }
        if let Some(ref s) = self.shape { w.write_with_tag(18, |w| w.write_message(s))?; }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum OneOfvalue<'a> {
    tensor_type(mod_TypeProto::Tensor<'a>),
    sequence_type(Box<mod_TypeProto::Sequence<'a>>),
    map_type(Box<mod_TypeProto::Map<'a>>),
    optional_type(Box<mod_TypeProto::Optional<'a>>),
    sparse_tensor_type(mod_TypeProto::SparseTensor<'a>),
    None,
}

impl<'a> Default for OneOfvalue<'a> {
    fn default() -> Self {
        OneOfvalue::None
    }
}

}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct OperatorSetIdProto<'a> {
    pub domain: Option<Cow<'a, str>>,
    pub version: Option<i64>,
}

impl<'a> MessageRead<'a> for OperatorSetIdProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.domain = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(16) => msg.version = Some(r.read_int64(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for OperatorSetIdProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.domain.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.version.as_ref().map_or(0, |m| 1 + sizeof_varint(*(m) as u64))
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.domain { w.write_with_tag(10, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.version { w.write_with_tag(16, |w| w.write_int64(*s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct FunctionProto<'a> {
    pub name: Option<Cow<'a, str>>,
    pub input: Vec<Cow<'a, str>>,
    pub output: Vec<Cow<'a, str>>,
    pub attribute: Vec<Cow<'a, str>>,
    pub attribute_proto: Vec<AttributeProto<'a>>,
    pub node: Vec<NodeProto<'a>>,
    pub doc_string: Option<Cow<'a, str>>,
    pub opset_import: Vec<OperatorSetIdProto<'a>>,
    pub domain: Option<Cow<'a, str>>,
    pub overload: Option<Cow<'a, str>>,
    pub value_info: Vec<ValueInfoProto<'a>>,
    pub metadata_props: Vec<StringStringEntryProto<'a>>,
}

impl<'a> MessageRead<'a> for FunctionProto<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.name = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(34) => msg.input.push(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(42) => msg.output.push(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(50) => msg.attribute.push(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(90) => msg.attribute_proto.push(r.read_message::<AttributeProto>(bytes)?),
                Ok(58) => msg.node.push(r.read_message::<NodeProto>(bytes)?),
                Ok(66) => msg.doc_string = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(74) => msg.opset_import.push(r.read_message::<OperatorSetIdProto>(bytes)?),
                Ok(82) => msg.domain = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(106) => msg.overload = Some(r.read_string(bytes).map(Cow::Borrowed)?),
                Ok(98) => msg.value_info.push(r.read_message::<ValueInfoProto>(bytes)?),
                Ok(114) => msg.metadata_props.push(r.read_message::<StringStringEntryProto>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for FunctionProto<'a> {
    fn get_size(&self) -> usize {
        0
        + self.name.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.input.iter().map(|s| 1 + sizeof_len((s).len())).sum::<usize>()
        + self.output.iter().map(|s| 1 + sizeof_len((s).len())).sum::<usize>()
        + self.attribute.iter().map(|s| 1 + sizeof_len((s).len())).sum::<usize>()
        + self.attribute_proto.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.node.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.doc_string.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.opset_import.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.domain.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.overload.as_ref().map_or(0, |m| 1 + sizeof_len((m).len()))
        + self.value_info.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
        + self.metadata_props.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if let Some(ref s) = self.name { w.write_with_tag(10, |w| w.write_string(&**s))?; }
        for s in &self.input { w.write_with_tag(34, |w| w.write_string(&**s))?; }
        for s in &self.output { w.write_with_tag(42, |w| w.write_string(&**s))?; }
        for s in &self.attribute { w.write_with_tag(50, |w| w.write_string(&**s))?; }
        for s in &self.attribute_proto { w.write_with_tag(90, |w| w.write_message(s))?; }
        for s in &self.node { w.write_with_tag(58, |w| w.write_message(s))?; }
        if let Some(ref s) = self.doc_string { w.write_with_tag(66, |w| w.write_string(&**s))?; }
        for s in &self.opset_import { w.write_with_tag(74, |w| w.write_message(s))?; }
        if let Some(ref s) = self.domain { w.write_with_tag(82, |w| w.write_string(&**s))?; }
        if let Some(ref s) = self.overload { w.write_with_tag(106, |w| w.write_string(&**s))?; }
        for s in &self.value_info { w.write_with_tag(98, |w| w.write_message(s))?; }
        for s in &self.metadata_props { w.write_with_tag(114, |w| w.write_message(s))?; }
        Ok(())
    }
}

