//! Owning RAII `Tensor` wrapper over PyTorch's `AtenTensorHandle` C ABI plus
//! `DType`, `Device`, and direct `libcudart` device/transfer helpers.

use std::ffi::{c_int, c_void};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::error::{require, Error, Result};
use crate::torch_sys::{
    self, check_aoti, check_cuda, shim_call, AtenTensorHandle, AtenTensorOpaque,
};

/// Scalar dtypes supported by `weights.safetensors` and model state buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    Float8E4M3Fn,
    BFloat16,
    Float16,
    Float32,
    Float64,
    Int64,
    Int32,
    Int16,
    Int8,
    UInt8,
    Bool,
}

impl DType {
    pub fn from_manifest_name(name: &str) -> Result<Self> {
        match name {
            "float8_e4m3fn" => Ok(Self::Float8E4M3Fn),
            "bfloat16" => Ok(Self::BFloat16),
            "float16" => Ok(Self::Float16),
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            "int64" => Ok(Self::Int64),
            "int32" => Ok(Self::Int32),
            "int16" => Ok(Self::Int16),
            "int8" => Ok(Self::Int8),
            "uint8" => Ok(Self::UInt8),
            "bool" => Ok(Self::Bool),
            _ => Err(Error::new(format!(
                "Shared weights dtype is unsupported: {name}"
            ))),
        }
    }

    /// Maps this runtime dtype to the corresponding `safetensors::Dtype` variant.
    pub fn to_safetensors(self) -> safetensors::Dtype {
        use safetensors::Dtype;
        match self {
            Self::Float8E4M3Fn => Dtype::F8_E4M3,
            Self::BFloat16 => Dtype::BF16,
            Self::Float16 => Dtype::F16,
            Self::Float32 => Dtype::F32,
            Self::Float64 => Dtype::F64,
            Self::Int64 => Dtype::I64,
            Self::Int32 => Dtype::I32,
            Self::Int16 => Dtype::I16,
            Self::Int8 => Dtype::I8,
            Self::UInt8 => Dtype::U8,
            Self::Bool => Dtype::BOOL,
        }
    }

    pub fn element_size(self) -> usize {
        match self {
            Self::Float8E4M3Fn | Self::Int8 | Self::UInt8 | Self::Bool => 1,
            Self::BFloat16 | Self::Float16 | Self::Int16 => 2,
            Self::Float32 | Self::Int32 => 4,
            Self::Float64 | Self::Int64 => 8,
        }
    }

    /// Queries LibTorch's runtime scalar-type code for this dtype.
    pub fn to_aoti(self) -> i32 {
        // SAFETY: `aoti_torch_dtype_*` are pure constant accessors.
        unsafe {
            match self {
                Self::Float8E4M3Fn => torch_sys::aoti_torch_dtype_float8_e4m3fn(),
                Self::BFloat16 => torch_sys::aoti_torch_dtype_bfloat16(),
                Self::Float16 => torch_sys::aoti_torch_dtype_float16(),
                Self::Float32 => torch_sys::aoti_torch_dtype_float32(),
                Self::Float64 => torch_sys::aoti_torch_dtype_float64(),
                Self::Int64 => torch_sys::aoti_torch_dtype_int64(),
                Self::Int32 => torch_sys::aoti_torch_dtype_int32(),
                Self::Int16 => torch_sys::aoti_torch_dtype_int16(),
                Self::Int8 => torch_sys::aoti_torch_dtype_int8(),
                Self::UInt8 => torch_sys::aoti_torch_dtype_uint8(),
                Self::Bool => torch_sys::aoti_torch_dtype_bool(),
            }
        }
    }

    pub fn from_aoti(code: i32) -> Option<Self> {
        let all = [
            Self::Float8E4M3Fn,
            Self::BFloat16,
            Self::Float16,
            Self::Float32,
            Self::Float64,
            Self::Int64,
            Self::Int32,
            Self::Int16,
            Self::Int8,
            Self::UInt8,
            Self::Bool,
        ];
        all.into_iter().find(|dt| dt.to_aoti() == code)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(i32),
}

impl Device {
    pub fn to_aoti(self) -> (i32, i32) {
        unsafe {
            match self {
                Self::Cpu => (torch_sys::aoti_torch_device_type_cpu(), -1),
                Self::Cuda(index) => (torch_sys::aoti_torch_device_type_cuda(), index),
            }
        }
    }

    pub fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda(_))
    }
}

/// Computes row-major (C-contiguous) element strides for `sizes`.
pub fn contiguous_strides(sizes: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; sizes.len()];
    let mut step = 1i64;
    for i in (0..sizes.len()).rev() {
        strides[i] = step;
        step = step.saturating_mul(sizes[i]);
    }
    strides
}

/// Queries `(major, minor)` SM compute capability for CUDA `device`.
pub fn cuda_device_capability(device: i32) -> Result<(i64, i64)> {
    let mut count: c_int = 0;
    let rc = unsafe { torch_sys::cudaGetDeviceCount(&mut count) };
    require(rc == 0 && count > device, "CUDA device 0 is required")?;

    let mut major: c_int = 0;
    let mut minor: c_int = 0;
    check_cuda(
        unsafe {
            torch_sys::cudaDeviceGetAttribute(
                &mut major,
                torch_sys::CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR,
                device,
            )
        },
        "cudaDeviceGetAttribute(major)",
    )?;
    check_cuda(
        unsafe {
            torch_sys::cudaDeviceGetAttribute(
                &mut minor,
                torch_sys::CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR,
                device,
            )
        },
        "cudaDeviceGetAttribute(minor)",
    )?;
    Ok((i64::from(major), i64::from(minor)))
}

/// Returns `(free_bytes, total_bytes)` on the active CUDA device.
pub fn cuda_mem_info() -> Result<(usize, usize)> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    check_cuda(
        unsafe { torch_sys::cudaMemGetInfo(&mut free, &mut total) },
        "cudaMemGetInfo",
    )?;
    Ok((free, total))
}

pub fn cuda_synchronize() -> Result<()> {
    check_cuda(
        unsafe { torch_sys::cudaDeviceSynchronize() },
        "cudaDeviceSynchronize",
    )
}

/// Returns PyTorch's borrowed current `cudaStream_t` for `device`.
pub fn current_cuda_stream(device: i32) -> Result<*mut c_void> {
    let mut stream: *mut c_void = std::ptr::null_mut();
    check_aoti(
        unsafe { torch_sys::aoti_torch_get_current_cuda_stream(device, &mut stream) },
        "aoti_torch_get_current_cuda_stream",
    )?;
    Ok(stream)
}

/// Owning RAII wrapper over an ATen tensor handle (`AtenTensorHandle`).
/// Deliberately `!Send` and `!Sync` to enforce single-thread GPU/stream ownership.
pub struct Tensor {
    handle: NonNull<AtenTensorOpaque>,
    _thread: PhantomData<Rc<()>>,
}

impl Tensor {
    /// Wraps a newly created owning `AtenTensorHandle`.
    ///
    /// # Safety
    /// `handle` must be a unique, owning pointer returned by PyTorch's AOTI C ABI
    /// (or null, which returns `Err`). Ownership transfers to the returned `Tensor`.
    pub(crate) unsafe fn from_raw(handle: AtenTensorHandle) -> Result<Self> {
        let handle = NonNull::new(handle).ok_or_else(|| Error::new("Null AtenTensorHandle"))?;
        Ok(Self {
            handle,
            _thread: PhantomData,
        })
    }

    pub fn raw(&self) -> AtenTensorHandle {
        self.handle.as_ptr()
    }

    pub fn empty_strided(
        sizes: &[i64],
        strides: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        require(
            sizes.len() == strides.len(),
            "Tensor sizes/strides rank mismatch",
        )?;
        let (device_type, device_index) = device.to_aoti();
        let mut handle: AtenTensorHandle = std::ptr::null_mut();
        check_aoti(
            unsafe {
                torch_sys::aoti_torch_empty_strided(
                    sizes.len() as i64,
                    sizes.as_ptr(),
                    strides.as_ptr(),
                    dtype.to_aoti(),
                    device_type,
                    device_index,
                    &mut handle,
                )
            },
            "aoti_torch_empty_strided",
        )?;
        unsafe { Self::from_raw(handle) }
    }

    pub fn empty_contiguous(sizes: &[i64], dtype: DType, device: Device) -> Result<Self> {
        let strides = contiguous_strides(sizes);
        Self::empty_strided(sizes, &strides, dtype, device)
    }

    /// Creates a zero-copy strided view sharing this tensor's storage.
    ///
    /// Note: `aoti_torch__reinterpret_tensor` applies `storage_offset` **relative**
    /// to `self.storage_offset()` (unlike `at::as_strided`).
    pub fn reinterpret(&self, sizes: &[i64], strides: &[i64], storage_offset: i64) -> Result<Self> {
        require(
            sizes.len() == strides.len(),
            "View sizes/strides rank mismatch",
        )?;
        let mut handle: AtenTensorHandle = std::ptr::null_mut();
        check_aoti(
            unsafe {
                torch_sys::aoti_torch__reinterpret_tensor(
                    self.raw(),
                    sizes.len() as i64,
                    sizes.as_ptr(),
                    strides.as_ptr(),
                    storage_offset,
                    &mut handle,
                )
            },
            "aoti_torch__reinterpret_tensor",
        )?;
        unsafe { Self::from_raw(handle) }
    }

    pub fn storage_offset(&self) -> Result<i64> {
        let mut offset: i64 = 0;
        check_aoti(
            unsafe { torch_sys::aoti_torch_get_storage_offset(self.raw(), &mut offset) },
            "aoti_torch_get_storage_offset",
        )?;
        Ok(offset)
    }

    /// Slices `dim` to `[start, end)` without copying data.
    pub fn slice_dim(&self, dim: usize, start: i64, end: i64) -> Result<Self> {
        let mut sizes = self.sizes()?;
        let strides = self.strides()?;
        require(
            dim < sizes.len() && start >= 0 && end >= start && end <= sizes[dim],
            "Invalid slice_dim bounds",
        )?;
        // aoti_torch__reinterpret_tensor applies `storage_offset` relative to `self.storage_offset()`.
        let delta = start
            .checked_mul(strides[dim])
            .ok_or_else(|| Error::new("slice_dim offset overflow"))?;
        sizes[dim] = end - start;
        self.reinterpret(&sizes, &strides, delta)
    }

    /// Narrows dimension 0 to `[start, start + length)` without copying data.
    pub fn narrow_dim0(&self, start: i64, length: i64) -> Result<Self> {
        require(length >= 0, "Invalid narrow_dim0 bounds")?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| Error::new("narrow_dim0 offset overflow"))?;
        self.slice_dim(0, start, end)
    }

    pub fn clone_tensor(&self) -> Result<Self> {
        let mut handle: AtenTensorHandle = std::ptr::null_mut();
        check_aoti(
            unsafe { torch_sys::aoti_torch_clone(self.raw(), &mut handle) },
            "aoti_torch_clone",
        )?;
        unsafe { Self::from_raw(handle) }
    }

    /// Return a contiguous host byte buffer containing the raw elements of this tensor.
    pub fn to_contiguous_host_bytes(&self) -> Result<Vec<u8>> {
        let count = usize::try_from(self.numel()?).map_err(|_| Error::new("Negative numel"))?;
        let nbytes = count
            .checked_mul(self.dtype()?.element_size())
            .ok_or_else(|| Error::new("Tensor byte size overflow"))?;
        if nbytes == 0 {
            return Ok(Vec::new());
        }

        let owned;
        let contig = if self.is_contiguous()? {
            self
        } else {
            let sizes = self.sizes()?;
            let mut tmp = Self::empty_contiguous(&sizes, self.dtype()?, self.device()?)?;
            tmp.copy_from(self)?;
            owned = tmp;
            &owned
        };

        let mut out = vec![0u8; nbytes];
        match contig.device()? {
            Device::Cuda(idx) => {
                let stream = current_cuda_stream(idx)?;
                let src = contig.data_ptr()?;
                check_cuda(
                    unsafe {
                        torch_sys::cudaMemcpyAsync(
                            out.as_mut_ptr().cast::<c_void>(),
                            src,
                            nbytes,
                            torch_sys::CUDA_MEMCPY_DEVICE_TO_HOST,
                            stream,
                        )
                    },
                    "cudaMemcpyAsync(D2H tensor)",
                )?;
                check_cuda(
                    unsafe { torch_sys::cudaStreamSynchronize(stream) },
                    "cudaStreamSynchronize(tensor)",
                )?;
            }
            Device::Cpu => {
                let src = contig.data_ptr()?.cast::<u8>();
                unsafe {
                    std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), nbytes);
                }
            }
        }
        Ok(out)
    }

    pub fn fill_scalar(&mut self, value: f64) -> Result<()> {
        match self.device()? {
            Device::Cpu => check_aoti(
                unsafe { torch_sys::aoti_torch_cpu_fill__Scalar(self.raw(), value) },
                "aoti_torch_cpu_fill__Scalar",
            ),
            Device::Cuda(_) => check_aoti(
                unsafe { torch_sys::aoti_torch_cuda_fill__Scalar(self.raw(), value) },
                "aoti_torch_cuda_fill__Scalar",
            ),
        }
    }

    /// Blocking copy from `src` into `self` (`non_blocking = 0`).
    pub fn copy_from(&mut self, src: &Tensor) -> Result<()> {
        // SAFETY: Both handles are live; `non_blocking = 0` synchronizes pageable H2D.
        check_aoti(
            unsafe { torch_sys::aoti_torch_copy_(self.raw(), src.raw(), 0) },
            "aoti_torch_copy_",
        )
    }

    /// Alignment-safe H2D byte copy into a contiguous CUDA tensor on PyTorch's current stream.
    pub fn copy_bytes_h2d_async(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let dev = match self.device()? {
            Device::Cuda(idx) => idx,
            Device::Cpu => return Err(Error::new("Expected CUDA destination tensor")),
        };
        let stream = current_cuda_stream(dev)?;
        let dst = self.data_ptr()?;
        check_cuda(
            unsafe {
                torch_sys::cudaMemcpyAsync(
                    dst,
                    bytes.as_ptr().cast::<c_void>(),
                    bytes.len(),
                    torch_sys::CUDA_MEMCPY_HOST_TO_DEVICE,
                    stream,
                )
            },
            "cudaMemcpyAsync(H2D)",
        )
    }

    /// Alignment-safe copy from host byte slice into a contiguous CPU tensor.
    pub fn copy_bytes_cpu(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        require(
            self.device()? == Device::Cpu,
            "Expected CPU destination tensor",
        )?;
        let dst = self.data_ptr()?.cast::<u8>();
        // SAFETY: `self` is a live contiguous CPU tensor sized for `bytes.len()` and does not alias `bytes`.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        Ok(())
    }

    pub fn data_ptr(&self) -> Result<*mut c_void> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check_aoti(
            unsafe { torch_sys::aoti_torch_get_data_ptr(self.raw(), &mut ptr) },
            "aoti_torch_get_data_ptr",
        )?;
        Ok(ptr)
    }

    pub fn dim(&self) -> Result<usize> {
        let mut d: i64 = 0;
        check_aoti(
            unsafe { torch_sys::aoti_torch_get_dim(self.raw(), &mut d) },
            "aoti_torch_get_dim",
        )?;
        usize::try_from(d).map_err(|_| Error::new("Negative tensor rank"))
    }

    pub fn numel(&self) -> Result<i64> {
        let mut n: i64 = 0;
        check_aoti(
            unsafe { torch_sys::aoti_torch_get_numel(self.raw(), &mut n) },
            "aoti_torch_get_numel",
        )?;
        Ok(n)
    }

    pub fn sizes(&self) -> Result<Vec<i64>> {
        let ndim = self.dim()?;
        if ndim == 0 {
            return Ok(Vec::new());
        }
        let mut ptr: *mut i64 = std::ptr::null_mut();
        check_aoti(
            unsafe { torch_sys::aoti_torch_get_sizes(self.raw(), &mut ptr) },
            "aoti_torch_get_sizes",
        )?;
        // SAFETY: `ptr` points to `ndim` valid `i64` entries owned by `self`.
        Ok(unsafe { std::slice::from_raw_parts(ptr, ndim) }.to_vec())
    }

    pub fn strides(&self) -> Result<Vec<i64>> {
        let ndim = self.dim()?;
        if ndim == 0 {
            return Ok(Vec::new());
        }
        let mut ptr: *mut i64 = std::ptr::null_mut();
        check_aoti(
            unsafe { torch_sys::aoti_torch_get_strides(self.raw(), &mut ptr) },
            "aoti_torch_get_strides",
        )?;
        // SAFETY: `ptr` points to `ndim` valid `i64` entries owned by `self`.
        Ok(unsafe { std::slice::from_raw_parts(ptr, ndim) }.to_vec())
    }

    pub fn dtype(&self) -> Result<DType> {
        let mut code: i32 = 0;
        check_aoti(
            unsafe { torch_sys::aoti_torch_get_dtype(self.raw(), &mut code) },
            "aoti_torch_get_dtype",
        )?;
        DType::from_aoti(code)
            .ok_or_else(|| Error::new(format!("Unsupported tensor dtype code: {code}")))
    }

    pub fn device(&self) -> Result<Device> {
        let mut dev_type: i32 = 0;
        check_aoti(
            unsafe { torch_sys::aoti_torch_get_device_type(self.raw(), &mut dev_type) },
            "aoti_torch_get_device_type",
        )?;
        let cpu_code = unsafe { torch_sys::aoti_torch_device_type_cpu() };
        let cuda_code = unsafe { torch_sys::aoti_torch_device_type_cuda() };
        if dev_type == cpu_code {
            Ok(Device::Cpu)
        } else if dev_type == cuda_code {
            let mut index: i32 = 0;
            check_aoti(
                unsafe { torch_sys::aoti_torch_get_device_index(self.raw(), &mut index) },
                "aoti_torch_get_device_index",
            )?;
            Ok(Device::Cuda(index))
        } else {
            Err(Error::new(format!(
                "Unsupported tensor device type: {dev_type}"
            )))
        }
    }

    pub fn is_contiguous(&self) -> Result<bool> {
        let mut flag = false;
        check_aoti(
            unsafe { torch_sys::aoti_torch_is_contiguous(self.raw(), &mut flag) },
            "aoti_torch_is_contiguous",
        )?;
        Ok(flag)
    }

    pub fn argmax_last_dim(&self) -> Result<i64> {
        let mut token: i64 = 0;
        shim_call(|err| unsafe { torch_sys::xinfer_shim_argmax(self.raw(), &mut token, err) })?;
        Ok(token)
    }

    /// Computes `argmax(-1)` on GPU for a 2D `[batch, vocab]` logits tensor and returns
    /// one host `i64` token ID per row.
    pub fn argmax_batch(&self) -> Result<Vec<i64>> {
        let sizes = self.sizes()?;
        require(sizes.len() == 2, "Expected 2D [batch, vocab] logits")?;
        let batch = usize::try_from(sizes[0])
            .map_err(|_| Error::invalid_input("Invalid logits batch size"))?;
        let mut tokens = vec![0i64; batch];
        shim_call(|err| unsafe {
            torch_sys::xinfer_shim_argmax_batch(self.raw(), tokens.as_mut_ptr(), batch, err)
        })?;
        Ok(tokens)
    }

    /// Copy contiguous BF16 CUDA/CPU logits to host FP32 (matching C++ `write_logits`).
    pub fn bf16_to_f32_host(&self) -> Result<Vec<f32>> {
        require(
            self.dtype()? == DType::BFloat16,
            "Expected BF16 logits tensor",
        )?;
        require(self.is_contiguous()?, "Expected contiguous logits tensor")?;
        let count = usize::try_from(self.numel()?).map_err(|_| Error::new("Negative numel"))?;
        let mut bf16 = vec![0u16; count];
        if count > 0 {
            let nbytes = count * std::mem::size_of::<u16>();
            match self.device()? {
                Device::Cuda(idx) => {
                    let stream = current_cuda_stream(idx)?;
                    let src = self.data_ptr()?;
                    check_cuda(
                        unsafe {
                            torch_sys::cudaMemcpyAsync(
                                bf16.as_mut_ptr().cast::<c_void>(),
                                src,
                                nbytes,
                                torch_sys::CUDA_MEMCPY_DEVICE_TO_HOST,
                                stream,
                            )
                        },
                        "cudaMemcpyAsync(D2H logits)",
                    )?;
                    check_cuda(
                        unsafe { torch_sys::cudaStreamSynchronize(stream) },
                        "cudaStreamSynchronize(logits)",
                    )?;
                }
                Device::Cpu => {
                    let src = self.data_ptr()?.cast::<u16>();
                    // SAFETY: `self` is contiguous BF16 on CPU with `count` elements.
                    unsafe {
                        std::ptr::copy_nonoverlapping(src, bf16.as_mut_ptr(), count);
                    }
                }
            }
        }
        // BF16 is the upper 16 bits of IEEE-754 FP32; widening on host avoids an extra GPU tensor.
        Ok(bf16
            .into_iter()
            .map(|bits| f32::from_bits(u32::from(bits) << 16))
            .collect())
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        // SAFETY: `self.handle` is uniquely owned by this `Tensor`.
        unsafe {
            let _ = torch_sys::aoti_torch_delete_tensor_object(self.handle.as_ptr());
        }
    }
}
