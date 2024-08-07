use crate::core::DeviceService;

// Todo:
/// The plan for the computation of the model instance.
///
/// This object informs the runtime on how to split the
/// memory and work needed by the model among all the devices
/// listed in the plan.
///
/// Concretely, this object describes:
/// - how the computational graph is partitioned
/// - the device that each subgraph should be executed on
/// - how to synchronize the execution of each sub-graph
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
