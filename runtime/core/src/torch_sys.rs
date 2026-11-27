//! Raw FFI declarations for `libxinfer_torch_shim.a`, PyTorch's `aoti_torch_*`
//! C ABI (`libtorch_cpu.so`/`libtorch_cuda.so`), and `libcudart`.

use std::ffi::{c_char, c_double, c_int, c_long, c_void, CStr};

use crate::error::{Error, Result};

/// Opaque handle matching PyTorch's `AtenTensorHandle` (`at::Tensor*`).
#[repr(C)]
pub struct AtenTensorOpaque {
    _private: [u8; 0],
}
pub type AtenTensorHandle = *mut AtenTensorOpaque;

/// Opaque handle owning one C++ `AOTIModelPackageLoader` instance.
#[repr(C)]
pub struct XinferPkg {
    _private: [u8; 0],
}

#[repr(C)]
pub struct XinferError {
    pub message: [c_char; 2048],
}

#[repr(C)]
pub struct XinferTorchInfo {
    pub torch_version: *const c_char,
    pub cudart_version: c_long,
    pub cxx11_abi: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct XinferSampleRow {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub uniform: f32,
}

pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 76;

extern "C" {
    // Static C++ shim (~111 LOC)
    pub fn xinfer_shim_torch_info(out: *mut XinferTorchInfo);
    pub fn xinfer_shim_last_torch_error() -> *const c_char;
    pub fn xinfer_shim_configure_runtime(error: *mut XinferError) -> c_int;
    pub fn xinfer_shim_pkg_open(
        pt2_path: *const c_char,
        out: *mut *mut XinferPkg,
        error: *mut XinferError,
    ) -> c_int;
    pub fn xinfer_shim_pkg_close(pkg: *mut XinferPkg);
    pub fn xinfer_shim_pkg_constant_fqns(
        pkg: *mut XinferPkg,
        callback: extern "C" fn(*const c_char, *mut c_void),
        ctx: *mut c_void,
        error: *mut XinferError,
    ) -> c_int;
    pub fn xinfer_shim_pkg_load_constants(
        pkg: *mut XinferPkg,
        names: *const *const c_char,
        handles: *const AtenTensorHandle,
        count: usize,
        error: *mut XinferError,
    ) -> c_int;
    pub fn xinfer_shim_pkg_run(
        pkg: *mut XinferPkg,
        inputs: *const AtenTensorHandle,
        num_inputs: usize,
        out_logits: *mut AtenTensorHandle,
        error: *mut XinferError,
    ) -> c_int;
    pub fn xinfer_shim_argmax(
        logits: AtenTensorHandle,
        token: *mut i64,
        error: *mut XinferError,
    ) -> c_int;
    pub fn xinfer_shim_argmax_batch(
        logits: AtenTensorHandle,
        tokens: *mut i64,
        count: usize,
        error: *mut XinferError,
    ) -> c_int;
    pub fn xinfer_shim_sample_batch(
        logits: AtenTensorHandle,
        params: *const XinferSampleRow,
        active: *const u8,
        out_tokens: *mut i64,
        count: usize,
        error: *mut XinferError,
    ) -> c_int;

    // PyTorch 2.13 aoti_torch_* C ABI (libtorch_cpu.so + libtorch_cuda.so)
    pub fn aoti_torch_device_type_cpu() -> i32;
    pub fn aoti_torch_device_type_cuda() -> i32;

    pub fn aoti_torch_dtype_float8_e4m3fn() -> i32;
    pub fn aoti_torch_dtype_bfloat16() -> i32;
    pub fn aoti_torch_dtype_float16() -> i32;
    pub fn aoti_torch_dtype_float32() -> i32;
    pub fn aoti_torch_dtype_float64() -> i32;
    pub fn aoti_torch_dtype_int64() -> i32;
    pub fn aoti_torch_dtype_int32() -> i32;
    pub fn aoti_torch_dtype_int16() -> i32;
    pub fn aoti_torch_dtype_int8() -> i32;
    pub fn aoti_torch_dtype_uint8() -> i32;
    pub fn aoti_torch_dtype_bool() -> i32;

    pub fn aoti_torch_empty_strided(
        ndim: i64,
        sizes_ptr: *const i64,
        strides_ptr: *const i64,
        dtype: i32,
        device_type: i32,
        device_index: i32,
        ret_new_tensor: *mut AtenTensorHandle,
    ) -> i32;
    pub fn aoti_torch__reinterpret_tensor(
        self_tensor: AtenTensorHandle,
        ndim: i64,
        sizes_ptr: *const i64,
        strides_ptr: *const i64,
        storage_offset: i64,
        ret_new_tensor: *mut AtenTensorHandle,
    ) -> i32;
    pub fn aoti_torch_delete_tensor_object(tensor: AtenTensorHandle) -> i32;
    pub fn aoti_torch_copy_(dst: AtenTensorHandle, src: AtenTensorHandle, non_blocking: i32)
        -> i32;
    pub fn aoti_torch_cpu_fill__Scalar(self_tensor: AtenTensorHandle, value: c_double) -> i32;
    pub fn aoti_torch_cuda_fill__Scalar(self_tensor: AtenTensorHandle, value: c_double) -> i32;

    pub fn aoti_torch_get_data_ptr(tensor: AtenTensorHandle, ret_data_ptr: *mut *mut c_void)
        -> i32;
    pub fn aoti_torch_get_dim(tensor: AtenTensorHandle, ret_dim: *mut i64) -> i32;
    pub fn aoti_torch_get_numel(tensor: AtenTensorHandle, ret_numel: *mut i64) -> i32;
    pub fn aoti_torch_get_sizes(tensor: AtenTensorHandle, ret_sizes: *mut *mut i64) -> i32;
    pub fn aoti_torch_get_strides(tensor: AtenTensorHandle, ret_strides: *mut *mut i64) -> i32;
    pub fn aoti_torch_get_storage_offset(
        tensor: AtenTensorHandle,
        ret_storage_offset: *mut i64,
    ) -> i32;
    pub fn aoti_torch_clone(tensor: AtenTensorHandle, ret_new_tensor: *mut AtenTensorHandle)
        -> i32;
    pub fn aoti_torch_get_dtype(tensor: AtenTensorHandle, ret_dtype: *mut i32) -> i32;
    pub fn aoti_torch_get_device_type(tensor: AtenTensorHandle, ret_device_type: *mut i32) -> i32;
    pub fn aoti_torch_get_device_index(tensor: AtenTensorHandle, ret_device_index: *mut i32)
        -> i32;
    pub fn aoti_torch_is_contiguous(tensor: AtenTensorHandle, ret_is_contiguous: *mut bool) -> i32;
    pub fn aoti_torch_get_current_cuda_stream(
        device_index: i32,
        ret_stream: *mut *mut c_void,
    ) -> i32;

    // Pure-C CUDA runtime API (libcudart)
    pub fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
    pub fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
    pub fn cudaDeviceSynchronize() -> c_int;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    pub fn cudaGetErrorString(error: c_int) -> *const c_char;
}

/// Invokes a C shim function following the `0 = ok, non-zero = XinferError` protocol.
pub fn shim_call(function: impl FnOnce(*mut XinferError) -> c_int) -> Result<()> {
    let mut error = XinferError { message: [0; 2048] };
    if function(&mut error) == 0 {
        Ok(())
    } else {
        // SAFETY: `error.message` is zero-initialized and the shim always NUL-terminates.
        let message = unsafe { CStr::from_ptr(error.message.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Err(Error::new(message))
    }
}

/// Converts an `aoti_torch_*` status code (`0 = ok`) into a `Result`, retrieving
/// PyTorch's thread-local AOTI error string on failure.
pub fn check_aoti(code: i32, context: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        // SAFETY: `xinfer_shim_last_torch_error` returns a valid NUL-terminated string.
        let detail = unsafe { CStr::from_ptr(xinfer_shim_last_torch_error()) }
            .to_string_lossy()
            .into_owned();
        if detail.is_empty() {
            Err(Error::new(format!("{context} failed (code {code})")))
        } else {
            Err(Error::new(format!("{context}: {detail}")))
        }
    }
}

/// Converts a `cudaError_t` (`0 = cudaSuccess`) into a `Result`.
pub fn check_cuda(code: c_int, context: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        // SAFETY: `cudaGetErrorString` always returns a static NUL-terminated C string.
        let detail = unsafe { CStr::from_ptr(cudaGetErrorString(code)) }
            .to_string_lossy()
            .into_owned();
        Err(Error::new(format!("{context}: {detail}")))
    }
}
