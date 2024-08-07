use crate::core::DeviceService;

pub struct Plan<D> {
    devices: Box<[D]>,
}

impl<D> Plan<D>
where
    D: DeviceService,
{
    pub fn new(devices: Box<[D]>) -> Self {
        Self { devices }
    }

    pub fn device(&self, id: usize) -> Option<&D> {
        self.devices.get(id)
    }
}
