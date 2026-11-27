//! Caller-owned in-place state buffer allocation, alignment, and reset policy.

use crate::error::{require, Result};
use crate::spec::StateSpec;
use crate::tensor::{Device, Tensor};

/// Caller-owned in-place state buffers for logits-only compiled packages.
pub struct StatePolicy {
    specs: Vec<StateSpec>,
    device: Device,
    allocated_batch_size: i64,
    tensors: Vec<Tensor>,
}

impl StatePolicy {
    pub fn new(
        specs: Vec<StateSpec>,
        device: Device,
        allocated_batch_size: Option<i64>,
    ) -> Result<Self> {
        require(!specs.is_empty(), "Empty state specification")?;
        require(
            !specs[0].shape.is_empty() && specs[0].shape[0] >= 1,
            "Invalid state batch shape",
        )?;
        let package_batch_size = specs[0].shape[0];
        let allocated_batch_size = allocated_batch_size.unwrap_or(package_batch_size);
        require(
            allocated_batch_size >= 1 && allocated_batch_size <= package_batch_size,
            "Invalid allocated batch size",
        )?;

        for item in &specs {
            require(
                !item.shape.is_empty() && item.shape[0] == package_batch_size,
                "Inconsistent state batch size",
            )?;
        }

        Ok(Self {
            specs,
            device,
            allocated_batch_size,
            tensors: Vec::new(),
        })
    }

    pub fn allocated_batch_size(&self) -> i64 {
        self.allocated_batch_size
    }

    /// Allocates state buffers on first call, then reuses them in place on subsequent
    /// resets. Fills KV buffers with `fill` and DeltaNet `conv`/`recurrent` buffers with 0.
    pub fn reset(&mut self, fill: f64) -> Result<()> {
        if self.tensors.is_empty() {
            let mut allocated = Vec::with_capacity(self.specs.len());
            for item in &self.specs {
                let mut shape = item.shape.clone();
                shape[0] = self.allocated_batch_size;
                let tensor = Tensor::empty_contiguous(&shape, item.dtype, self.device)?;
                allocated.push(tensor);
            }
            self.tensors = allocated;
        }

        for (item, tensor) in self.specs.iter().zip(self.tensors.iter_mut()) {
            // Keep base pointers 16-byte aligned so AOTI's alignment guards never clone inputs.
            let addr = tensor.data_ptr()? as usize;
            require((addr & 15) == 0, "State buffer must be 16-byte aligned")?;
            tensor.fill_scalar(if item.is_kv() { fill } else { 0.0 })?;
        }
        Ok(())
    }

    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    pub fn tensors_mut(&mut self) -> &mut [Tensor] {
        &mut self.tensors
    }

    /// Validates the logits-only in-place state post-step ABI contract.
    pub fn accept(&self, num_outputs: usize, batch_size: i64) -> Result<()> {
        require(
            !self.tensors.is_empty(),
            "State must be reset before execution",
        )?;
        require(num_outputs == 1, "Incorrect package output count")?;
        require(
            batch_size >= 1 && batch_size <= self.allocated_batch_size,
            "Invalid state batch size",
        )
    }
}
