/// The provider allocates and manages device memory.
pub trait Provider: Clone + Sized {
    type Data;
    type Device;
    /// Allocates a tensor.
    fn device(&self) -> Self::Device;
}
