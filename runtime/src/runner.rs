//! AOTI package lifecycle, coalesced H2D staging runner, prompt chunking,
//! greedy generation loop, and the public `Model` / `Session` API.

use std::ffi::{c_char, c_void, CStr, CString};
use std::io::Write;
use std::marker::PhantomData;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::error::{require, Error, Result};
use crate::spec::{configured_device, Spec};
use crate::state::StatePolicy;
use crate::tensor::{cuda_synchronize, DType, Device, Tensor};
use crate::torch_sys::{self, shim_call, AtenTensorHandle, XinferPkg};
use crate::weights::{SharedWeights, ValidatedBundle};

/// Owning RAII handle to an AOTI compiled package (`AOTIModelPackageLoader`).
/// Deliberately `!Send` and `!Sync`.
pub struct Package {
    handle: NonNull<XinferPkg>,
    _thread: PhantomData<Rc<()>>,
}

impl Package {
    pub fn open(pt2_path: &Path) -> Result<Self> {
        let c_path = CString::new(pt2_path.as_os_str().as_bytes())?;
        let mut raw: *mut XinferPkg = std::ptr::null_mut();
        shim_call(|err| unsafe {
            torch_sys::xinfer_shim_pkg_open(c_path.as_ptr(), &mut raw, err)
        })?;
        let handle = NonNull::new(raw).ok_or_else(|| Error::new("Null XinferPkg handle"))?;
        Ok(Self {
            handle,
            _thread: PhantomData,
        })
    }

    pub fn constant_fqns(&self) -> Result<Vec<String>> {
        extern "C" fn collect(fqn: *const c_char, ctx: *mut c_void) {
            if fqn.is_null() || ctx.is_null() {
                return;
            }
            // SAFETY: `ctx` points to our local `Vec<String>` and `fqn` is a valid C string
            // for the duration of `xinfer_shim_pkg_constant_fqns`.
            let out = unsafe { &mut *(ctx as *mut Vec<String>) };
            let text = unsafe { CStr::from_ptr(fqn) }
                .to_string_lossy()
                .into_owned();
            out.push(text);
        }

        let mut names = Vec::<String>::new();
        shim_call(|err| unsafe {
            torch_sys::xinfer_shim_pkg_constant_fqns(
                self.handle.as_ptr(),
                collect,
                (&mut names as *mut Vec<String>).cast::<c_void>(),
                err,
            )
        })?;
        Ok(names)
    }

    pub fn load_constants(&mut self, constants: &[(&str, &Tensor)]) -> Result<()> {
        let mut c_names = Vec::with_capacity(constants.len());
        let mut name_ptrs = Vec::with_capacity(constants.len());
        let mut handles = Vec::with_capacity(constants.len());
        for &(name, tensor) in constants {
            let c_name = CString::new(name)?;
            name_ptrs.push(c_name.as_ptr());
            c_names.push(c_name);
            handles.push(tensor.raw());
        }
        shim_call(|err| unsafe {
            torch_sys::xinfer_shim_pkg_load_constants(
                self.handle.as_ptr(),
                name_ptrs.as_ptr(),
                handles.as_ptr(),
                handles.len(),
                err,
            )
        })
    }

    pub fn run(&mut self, inputs: &[Tensor]) -> Result<Tensor> {
        let handles: Vec<AtenTensorHandle> = inputs.iter().map(Tensor::raw).collect();
        let mut out_raw: AtenTensorHandle = std::ptr::null_mut();
        shim_call(|err| unsafe {
            torch_sys::xinfer_shim_pkg_run(
                self.handle.as_ptr(),
                handles.as_ptr(),
                handles.len(),
                &mut out_raw,
                err,
            )
        })?;
        // SAFETY: `xinfer_shim_pkg_run` transfers ownership of `out_raw` on success.
        unsafe { Tensor::from_raw(out_raw) }
    }
}

impl Drop for Package {
    fn drop(&mut self) {
        // SAFETY: `self.handle` is uniquely owned by this `Package`.
        unsafe {
            torch_sys::xinfer_shim_pkg_close(self.handle.as_ptr());
        }
    }
}

/// Abstract step driver enabling CPU-only unit tests of prompt chunking and greedy generation.
pub trait StepDriver {
    type Logits;

    fn reset(&mut self) -> Result<()>;
    fn step(&mut self, tokens: &[i64]) -> Result<Self::Logits>;
    fn prefill(&mut self, prompt: &[i64]) -> Result<Self::Logits>;
    fn argmax(&mut self, logits: &Self::Logits) -> Result<i64>;
    fn write_trace(&mut self, logits: &Self::Logits, writer: &mut dyn Write) -> Result<()>;
}

/// Request length and compiled-call length are separate contracts. A final singleton
/// goes through `step()`'s decode path, with all earlier state carried.
pub fn prefill_with<D: StepDriver>(
    driver: &mut D,
    prompt: &[i64],
    max_prefill: i64,
) -> Result<D::Logits> {
    require(
        !prompt.is_empty() && max_prefill >= 2,
        "Invalid prompt partition",
    )?;
    let chunk_size = max_prefill as usize;
    let mut logits = None;
    let mut start = 0;
    while start < prompt.len() {
        let count = (prompt.len() - start).min(chunk_size);
        logits = Some(driver.step(&prompt[start..start + count])?);
        start += count;
    }
    logits.ok_or_else(|| Error::new("Invalid prompt partition"))
}

pub fn generate_with<D: StepDriver>(
    driver: &mut D,
    prompt: &[i64],
    limit: i64,
    eos: &[i64],
    mut trace: Option<&mut dyn Write>,
) -> Result<Vec<i64>> {
    driver.reset()?;
    let mut logits = driver.prefill(prompt)?;

    let mut generated = Vec::with_capacity(limit.max(0) as usize);
    for step_idx in 0..limit {
        if let Some(writer) = trace.as_deref_mut() {
            driver.write_trace(&logits, writer)?;
        }

        let token = driver.argmax(&logits)?;
        generated.push(token);
        if eos.contains(&token) || step_idx + 1 == limit {
            break;
        }
        logits = driver.step(&[token])?;
    }
    Ok(generated)
}

/// Widens contiguous BF16 logits to little-endian FP32 and writes them to `writer`.
pub fn write_logits_f32(writer: &mut dyn Write, logits: &Tensor) -> Result<()> {
    let values = logits.bf16_to_f32_host()?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    writer
        .write_all(&bytes)
        .map_err(|_| Error::new("Failed to write logits trace"))
}

/// Core Rust runtime executing prefill/decode AOTI packages with caller-owned state and
/// coalesced H2D input staging.
///
/// Field declaration order is load-bearing: Rust drops fields in declaration order,
/// so borrowing `prefill` and `decode` packages MUST be declared before `_weights`.
pub struct Runner {
    spec: Spec,
    allocated_batch_size: i64,
    device: Device,
    // Borrowing loaders MUST drop before canonical `_weights`.
    prefill: Package,
    decode: Package,
    _weights: SharedWeights,
    state: StatePolicy,
    host_token_pos: Tensor,
    device_token_pos: Tensor,
    host_lengths: Tensor,
    positions: Vec<i64>,
    position: i64,
    prefill_calls: i64,
    decode_calls: i64,
}

impl Runner {
    fn resolve_allocated_batch_size(spec: &Spec, allocated_batch_size: Option<i64>) -> Result<i64> {
        let batch_size = allocated_batch_size.unwrap_or(spec.max_batch_size);
        require(
            batch_size >= 1 && batch_size <= spec.max_batch_size,
            "Invalid allocated batch size",
        )?;
        Ok(batch_size)
    }

    pub fn open(packages: &Path, spec: Spec, allocated_batch_size: Option<i64>) -> Result<Self> {
        let allocated_batch_size = Self::resolve_allocated_batch_size(&spec, allocated_batch_size)?;
        let bundle = spec.validate_files(packages)?;
        Self::from_validated(spec, bundle, allocated_batch_size)
    }

    fn from_validated(
        spec: Spec,
        bundle: ValidatedBundle,
        allocated_batch_size: i64,
    ) -> Result<Self> {
        let device = configured_device(&spec)?;
        let weights = SharedWeights::load(bundle, &spec.shared_weights)?;
        let mut prefill = Package::open(&weights.package_path("prefill")?)?;
        let mut decode = Package::open(&weights.package_path("decode")?)?;
        let state = StatePolicy::new(spec.state.clone(), device, Some(allocated_batch_size))?;

        // Reserve space for both `input_ids` and 16-byte-aligned `cache_positions` in one buffer.
        let staging_len = 2 * ((allocated_batch_size * spec.max_prefill + 1) & !1i64);
        let host_token_pos = Tensor::empty_contiguous(&[staging_len], DType::Int64, Device::Cpu)?;
        let device_token_pos = Tensor::empty_contiguous(&[staging_len], DType::Int64, device)?;
        let host_lengths =
            Tensor::empty_contiguous(&[2, allocated_batch_size], DType::Int32, Device::Cpu)?;

        weights.bind_to_package(&mut prefill, "prefill")?;
        weights.bind_to_package(&mut decode, "decode")?;

        let mut runner = Self {
            spec,
            allocated_batch_size,
            device,
            prefill,
            decode,
            _weights: weights,
            state,
            host_token_pos,
            device_token_pos,
            host_lengths,
            positions: vec![0; allocated_batch_size as usize],
            position: 0,
            prefill_calls: 0,
            decode_calls: 0,
        };
        runner.reset(0.0)?;
        Ok(runner)
    }

    pub fn reset(&mut self, fill: f64) -> Result<()> {
        self.state.reset(fill)?;
        self.position = 0;
        self.prefill_calls = 0;
        self.decode_calls = 0;
        self.positions.fill(0);
        Ok(())
    }

    pub fn step(&mut self, tokens: &[i64]) -> Result<Tensor> {
        self.step_batch(&[tokens])
    }

    /// Executes one compiled `decode` (max row len == 1) or `prefill` (len in 2..=512)
    /// call for `1..=allocated_batch_size` rows. Empty rows remain inactive in place.
    pub fn step_batch(&mut self, rows: &[&[i64]]) -> Result<Tensor> {
        let length = self.spec.check_batch_tokens(rows)?;
        let batch_size = rows.len() as i64;
        require(
            batch_size <= self.allocated_batch_size,
            "Batch size exceeds allocated state",
        )?;
        for (row_idx, row) in rows.iter().enumerate() {
            let row_len = row.len() as i64;
            require(
                row_len <= self.spec.capacity - self.positions[row_idx],
                "Context capacity exceeded",
            )?;
            require(
                self.positions[row_idx] + row_len > 0,
                "Each row must start with tokens",
            )?;
        }

        let count = batch_size * length;
        // Keep cache_positions 16-byte aligned for AOTI's compiled input alignment.
        let stride = (count + 1) & !1i64;
        let total = (stride + count) as usize;
        let length_usize = length as usize;
        let alloc_batch_usize = self.allocated_batch_size as usize;

        {
            let token_pos_ptr = self.host_token_pos.data_ptr()?.cast::<i64>();
            // SAFETY: `host_token_pos` is an exclusively owned CPU `Int64` tensor of length `>= total`.
            let token_pos = unsafe { std::slice::from_raw_parts_mut(token_pos_ptr, total) };
            token_pos.fill(0);

            let lengths_ptr = self.host_lengths.data_ptr()?.cast::<i32>();
            // SAFETY: `host_lengths` is an exclusively owned CPU `Int32` tensor of `[2, alloc_batch]`.
            let lengths =
                unsafe { std::slice::from_raw_parts_mut(lengths_ptr, 2 * alloc_batch_usize) };
            let (kv_data, query_data) = lengths.split_at_mut(alloc_batch_usize);

            let stride_usize = stride as usize;
            for (row_idx, row) in rows.iter().enumerate() {
                let base_pos = self.positions[row_idx];
                for (tok_idx, &token) in row.iter().enumerate() {
                    token_pos[row_idx * length_usize + tok_idx] = token;
                    token_pos[stride_usize + row_idx * length_usize + tok_idx] =
                        base_pos + tok_idx as i64;
                }
                let row_tokens = row.len() as i64;
                kv_data[row_idx] = (base_pos + row_tokens) as i32;
                query_data[row_idx] = row_tokens as i32;
            }
        }

        // One coalesced pageable H2D copy avoids stalling the GPU stream while
        // keeping staging buffers safe to reuse on the next step_batch call.
        let mut staged = self.device_token_pos.narrow_dim0(0, stride + count)?;
        let host_staged = self.host_token_pos.narrow_dim0(0, stride + count)?;
        staged.copy_from(&host_staged)?;

        let mut inputs = Vec::with_capacity(self.state.tensors().len() + 4);
        inputs.push(
            self.device_token_pos
                .reinterpret(&[batch_size, length], &[length, 1], 0)?,
        );
        inputs.push(self.device_token_pos.reinterpret(
            &[batch_size, length],
            &[length, 1],
            stride,
        )?);
        // Prefix views preserve the allocation and unused rows' state.
        for tensor in self.state.tensors() {
            inputs.push(tensor.narrow_dim0(0, batch_size)?);
        }
        inputs.push(self.host_lengths.reinterpret(&[batch_size], &[1], 0)?);
        inputs.push(self.host_lengths.reinterpret(
            &[batch_size],
            &[1],
            self.allocated_batch_size,
        )?);

        let output = self.run_inputs(&inputs, length, batch_size)?;
        drop(inputs);

        for (row_idx, row) in rows.iter().enumerate() {
            self.positions[row_idx] += row.len() as i64;
        }
        self.position = self.positions.iter().copied().max().unwrap_or(0);
        Ok(output)
    }

    fn run_inputs(&mut self, inputs: &[Tensor], length: i64, batch_size: i64) -> Result<Tensor> {
        let output = if length == 1 {
            self.decode.run(inputs)?
        } else {
            self.prefill.run(inputs)?
        };
        require(
            output.sizes()? == [batch_size, self.spec.vocab]
                && output.dtype()? == DType::BFloat16
                && output.device()? == self.device,
            "Unexpected compiled output",
        )?;
        self.state.accept(1, batch_size)?;
        if length == 1 {
            self.decode_calls += 1;
        } else {
            self.prefill_calls += 1;
        }
        Ok(output)
    }

    pub fn prefill(&mut self, prompt: &[i64]) -> Result<Tensor> {
        // Validate the complete request before its first chunk can change state.
        self.spec.check_tokens(prompt)?;
        require(
            (prompt.len() as i64) <= self.spec.capacity - self.positions[0],
            "Context capacity exceeded",
        )?;
        let max_prefill = self.spec.max_prefill;
        prefill_with(self, prompt, max_prefill)
    }

    pub fn generate(
        &mut self,
        prompt: &[i64],
        limit: i64,
        trace: Option<&mut dyn Write>,
    ) -> Result<Vec<i64>> {
        let eos = self.spec.eos.clone();
        self.generate_with_eos(prompt, limit, &eos, trace)
    }

    pub fn generate_with_eos(
        &mut self,
        prompt: &[i64],
        limit: i64,
        eos: &[i64],
        trace: Option<&mut dyn Write>,
    ) -> Result<Vec<i64>> {
        self.spec.check_request(prompt, limit)?;
        generate_with(self, prompt, limit, eos, trace)
    }

    pub fn spec(&self) -> &Spec {
        &self.spec
    }

    pub fn allocated_batch_size(&self) -> i64 {
        self.allocated_batch_size
    }

    pub fn position(&self) -> i64 {
        self.position
    }

    pub fn positions(&self) -> &[i64] {
        &self.positions
    }

    pub fn prefill_calls(&self) -> i64 {
        self.prefill_calls
    }

    pub fn decode_calls(&self) -> i64 {
        self.decode_calls
    }

    pub fn state(&self) -> &[Tensor] {
        self.state.tensors()
    }
}

impl StepDriver for Runner {
    type Logits = Tensor;

    fn reset(&mut self) -> Result<()> {
        Runner::reset(self, 0.0)
    }

    fn step(&mut self, tokens: &[i64]) -> Result<Tensor> {
        Runner::step(self, tokens)
    }

    fn prefill(&mut self, prompt: &[i64]) -> Result<Tensor> {
        Runner::prefill(self, prompt)
    }

    fn argmax(&mut self, logits: &Tensor) -> Result<i64> {
        logits.argmax_last_dim()
    }

    fn write_trace(&mut self, logits: &Tensor, writer: &mut dyn Write) -> Result<()> {
        write_logits_f32(writer, logits)
    }
}

/// High-level validated package handle. CPU-only until `open_session` is called.
pub struct Model {
    packages: std::path::PathBuf,
    spec: Spec,
}

impl Model {
    pub fn open(packages: impl AsRef<Path>) -> Result<Self> {
        let packages = packages.as_ref().to_path_buf();
        let spec = Spec::load(&packages)?;
        Ok(Self { packages, spec })
    }

    pub fn spec(&self) -> &Spec {
        &self.spec
    }

    /// Open a GPU execution session.
    ///
    /// `Session` is deliberately `!Send` and `!Sync` to enforce single-thread GPU/stream ownership:
    ///
    /// ```compile_fail
    /// fn assert_send<T: Send>() {}
    /// assert_send::<xinfer_runtime::Session>();
    /// ```
    ///
    /// ```compile_fail
    /// fn assert_sync<T: Sync>() {}
    /// assert_sync::<xinfer_runtime::Session>();
    /// ```
    pub fn open_session(&self, allocated_batch_size: Option<i64>) -> Result<Session> {
        let runner = Runner::open(&self.packages, self.spec.clone(), allocated_batch_size)?;
        Ok(Session { runner })
    }
}

/// Owning single-thread GPU inference session (`!Send + !Sync`).
pub struct Session {
    runner: Runner,
}

impl Session {
    pub fn spec(&self) -> &Spec {
        self.runner.spec()
    }

    pub fn allocated_batch_size(&self) -> i64 {
        self.runner.allocated_batch_size()
    }

    pub fn reset(&mut self, fill: f64) -> Result<()> {
        self.runner.reset(fill)
    }

    pub fn step(&mut self, tokens: &[i64]) -> Result<Tensor> {
        self.runner.step(tokens)
    }

    pub fn step_batch(&mut self, rows: &[&[i64]]) -> Result<Tensor> {
        self.runner.step_batch(rows)
    }

    pub fn prefill(&mut self, prompt: &[i64]) -> Result<Tensor> {
        self.runner.prefill(prompt)
    }

    pub fn generate(
        &mut self,
        prompt: &[i64],
        limit: i64,
        trace: Option<&mut dyn Write>,
    ) -> Result<Vec<i64>> {
        self.runner.generate(prompt, limit, trace)
    }

    pub fn generate_with_eos(
        &mut self,
        prompt: &[i64],
        limit: i64,
        eos: &[i64],
        trace: Option<&mut dyn Write>,
    ) -> Result<Vec<i64>> {
        self.runner.generate_with_eos(prompt, limit, eos, trace)
    }

    pub fn position(&self) -> i64 {
        self.runner.position()
    }

    pub fn positions(&self) -> &[i64] {
        self.runner.positions()
    }

    pub fn prefill_calls(&self) -> i64 {
        self.runner.prefill_calls()
    }

    pub fn decode_calls(&self) -> i64 {
        self.runner.decode_calls()
    }

    pub fn state(&self) -> &[Tensor] {
        self.runner.state()
    }

    pub fn synchronize(&self) -> Result<()> {
        cuda_synchronize()
    }
}
