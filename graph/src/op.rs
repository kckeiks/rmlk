use std::str::FromStr;

#[derive(Clone, Copy)]
pub enum Op {
    NoOp,
    Add,
    Mul,
    MatMul,
    Sub,
}

impl FromStr for Op {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let op = match s {
            "add" => Self::Add,
            _ => return Err(()),
        };
        Ok(op)
    }
}
