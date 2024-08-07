use crate::core::ExecutionProvider;

pub struct Plan<P> {
    providers: Box<[P]>,
}

impl<P> Plan<P>
where
    P: ExecutionProvider,
{
    pub fn new(providers: Box<[P]>) -> Self {
        Self { providers }
    }

    pub fn provider(&self, id: usize) -> Option<&P> {
        self.providers.get(id)
    }
}
