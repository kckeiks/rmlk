use anyhow::Result;
use log::debug;
use rmlk_schema::Attribute;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub struct GemmAttributes {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

impl GemmAttributes {
    pub fn new(attrs: &HashMap<Box<str>, Attribute>) -> Result<Self> {
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
                Some(n) => {
                    debug!("invalid value {n} for the `transA` attribute");
                    return Err(Box::new(GemmAttributeError::InvalidTransA).into());
                }
            };
        }

        if let Some(attr) = attrs.get("transB") {
            trans_b = match attr.int() {
                Some(0) => Some(false),
                Some(1) => Some(true),
                None => None,
                Some(n) => {
                    debug!("invalid value {n} for the `transB` attribute");
                    return Err(Box::new(GemmAttributeError::InvalidTransB).into());
                }
            };
        }

        Ok(Self {
            alpha: alpha.unwrap_or(1.0),
            beta: beta.unwrap_or(1.0),
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

impl Default for GemmAttributes {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
        }
    }
}

#[derive(Debug)]
pub enum GemmAttributeError {
    InvalidTransA,
    InvalidTransB,
}

impl Display for GemmAttributeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for GemmAttributeError {}
