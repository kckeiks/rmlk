use crate::Error;
use rmlk_ir::Attribute;
use std::collections::HashMap;

#[derive(Debug)]
pub struct GemmAttributes {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

impl GemmAttributes {
    pub fn new(attrs: &HashMap<Box<str>, Attribute>) -> crate::Result<Self> {
        let mut alpha = None;
        let mut beta = None;
        let mut trans_a = None;
        let mut trans_b = None;

        if let Some(attr) = attrs.get("alpha") {
            alpha = attr.float();
        }

        if let Some(attr) = attrs.get("beta") {
            beta = attr.float();
        }

        if let Some(attr) = attrs.get("transA") {
            trans_a = match attr.int() {
                Some(0) => Some(false),
                Some(1) => Some(true),
                None => None,
                _ => return Err(Error::InvalidAttribute),
            };
        }

        if let Some(attr) = attrs.get("transB") {
            trans_b = match attr.int() {
                Some(0) => Some(false),
                Some(1) => Some(true),
                None => None,
                _ => return Err(Error::InvalidAttribute),
            };
        }

        Ok(Self {
            alpha: alpha.unwrap_or(1.0),
            beta: beta.unwrap_or(0.0),
            trans_a: trans_a.unwrap_or(false),
            trans_b: trans_b.unwrap_or(false),
        })
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn beta(&self) -> f32 {
        self.beta
    }

    pub fn trans_a(&self) -> bool {
        self.trans_a
    }

    pub fn trans_b(&self) -> bool {
        self.trans_b
    }
}
