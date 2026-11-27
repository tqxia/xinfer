//! Bundle checksum verification, `safetensors` archive parsing, POSIX `mmap` loading,
//! and shared zero-copy weight binding across `prefill.pt2` and `decode.pt2`.

use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use safetensors::tensor::Metadata as SafetensorsMetadata;
use safetensors::SafeTensors;
use serde::Deserialize;
use serde_json::Value;

use crate::error::{require, Error, Result};
use crate::runner::Package;
use crate::sha256::sha256_file;
use crate::tensor::{cuda_synchronize, DType, Device, Tensor};

pub const SHARED_WEIGHTS_FILE: &str = "weights.safetensors";
pub const SHARED_WEIGHTS_MANIFEST_KEY: &str = "xinfer_manifest";

/// Embedded JSON manifest stored under `__metadata__.xinfer_manifest` inside `weights.safetensors`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddedManifest {
    identity: Value,
    identity_sha256: String,
    entries: HashMap<String, ManifestEntry>,
    #[allow(dead_code)]
    sources: Value,
    storages: HashMap<String, StorageDescriptor>,
    tensors: HashMap<String, TensorDescriptor>,
}

#[derive(Debug, Deserialize)]
struct ManifestEntry {
    package_sha256: String,
    constants: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct StorageDescriptor {
    dtype: String,
    device: String,
    bytes: u64,
}

#[derive(Debug, Deserialize)]
struct TensorDescriptor {
    storage: String,
    dtype: String,
    device: String,
    shape: Vec<i64>,
    stride: Vec<i64>,
    offset: i64,
}

/// Read-only RAII `mmap(MAP_PRIVATE)` guard that unmaps backing pages on `Drop`.
struct MappedFile {
    ptr: *const u8,
    len: usize,
}

impl MappedFile {
    fn open(path: &Path, expected_size: u64) -> Result<Self> {
        if expected_size == 0 {
            return Ok(Self {
                ptr: std::ptr::null(),
                len: 0,
            });
        }
        let len = usize::try_from(expected_size)
            .map_err(|_| Error::new("Shared weights archive size exceeds address space"))?;
        let c_path = CString::new(path.as_os_str().as_bytes())?;
        // SAFETY: `c_path` is a valid NUL-terminated path string.
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY) };
        if fd < 0 {
            return Err(Error::new(format!(
                "Cannot open shared weights archive: {}",
                path.display()
            )));
        }

        let mut stat = unsafe { std::mem::zeroed::<libc::stat>() };
        let stat_ok = unsafe { libc::fstat(fd, &mut stat) } == 0
            && stat.st_size >= 0
            && (stat.st_size as u64) == expected_size;
        if !stat_ok {
            unsafe {
                libc::close(fd);
            }
            return Err(Error::new(
                "Shared weights archive size changed during load",
            ));
        }

        // SAFETY: `fd` is an open read-only descriptor whose size was verified to equal `len`.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            )
        };
        unsafe {
            libc::close(fd);
        }
        if mapped == libc::MAP_FAILED {
            return Err(Error::new(format!(
                "Cannot map shared weights archive: {}",
                path.display()
            )));
        }
        Ok(Self {
            ptr: mapped.cast::<u8>(),
            len,
        })
    }

    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: `self.ptr` is a valid read-only mapping of `self.len` bytes for `&self`'s lifetime.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            // SAFETY: `self.ptr` / `self.len` came from a successful `libc::mmap` call.
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}

/// Validates and parses the safetensors metadata header of `path` using `safetensors::SafeTensors`.
///
/// Demand-paged `MAP_PRIVATE` mapping only faults the header pages while verifying
/// header length, JSON schema, tensor offsets, and total file length.
pub fn read_safetensors_metadata(path: &Path) -> Result<(u64, SafetensorsMetadata)> {
    let file_size = fs::metadata(path)
        .map_err(|_| Error::new("Cannot open safetensors archive"))?
        .len();
    require(file_size >= 8, "Truncated safetensors header")?;
    let mapped = MappedFile::open(path, file_size)?;
    let (_, metadata) = SafeTensors::read_metadata(mapped.as_slice())
        .map_err(|err| Error::new(format!("Invalid safetensors archive: {err}")))?;
    Ok((file_size, metadata))
}

/// Follows symlinks (matching C++ `fs::is_regular_file`) so symlinked bundle fixtures work.
fn is_regular_file(path: &Path) -> bool {
    fs::metadata(path).map(|m| m.is_file()).unwrap_or(false)
}

/// A load-scoped proof: only verification can bind a manifest to package paths.
pub struct ValidatedBundle {
    directory: PathBuf,
    contract: Value,
    manifest: EmbeddedManifest,
    safetensors_metadata: SafetensorsMetadata,
    file_size: u64,
}

impl ValidatedBundle {
    pub fn verify(directory: &Path, metadata: &Value) -> Result<Self> {
        let root = fs::canonicalize(directory).map_err(|_| {
            Error::new(format!(
                "Cannot resolve package directory: {}",
                directory.display()
            ))
        })?;
        let declared_entries = metadata
            .get("entries")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::new("Unsupported package metadata: entries"))?;

        let mut package_digests = HashMap::new();
        for kind in ["prefill", "decode"] {
            let entry = declared_entries
                .get(kind)
                .and_then(Value::as_object)
                .ok_or_else(|| Error::new("Unsupported package metadata: entries"))?;
            let expected_bytes = entry
                .get("bytes")
                .and_then(Value::as_u64)
                .ok_or_else(|| Error::new("Unsupported package metadata: entries"))?;
            let expected_sha = entry
                .get("sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::new("Unsupported package metadata: entries"))?;

            let path = root.join(format!("{kind}.pt2"));
            require(
                is_regular_file(&path),
                format!("Package file is missing: {kind}"),
            )?;
            let actual_bytes = fs::metadata(&path)?.len();
            require(actual_bytes == expected_bytes, "Package file size mismatch")?;
            let digest = sha256_file(&path)?;
            require(
                digest == expected_sha,
                format!("Package checksum mismatch: {kind}"),
            )?;
            package_digests.insert(kind, digest);
        }

        let contract = metadata.get("shared_weights").ok_or_else(|| {
            Error::new("Shared weights metadata is required; re-export both packages")
        })?;
        let contract_obj = contract.as_object().ok_or_else(|| {
            Error::new("Shared weights metadata is required; re-export both packages")
        })?;
        let contract_valid = contract_obj.len() == 3
            && contract_obj
                .get("identity")
                .and_then(Value::as_str)
                .is_some()
            && contract_obj
                .get("bytes")
                .and_then(Value::as_i64)
                .map(|b| b >= 0)
                .unwrap_or(false)
            && contract_obj.get("sha256").and_then(Value::as_str).is_some();
        require(
            contract_valid,
            "Shared weights metadata is required; re-export both packages",
        )?;

        let path = root.join(SHARED_WEIGHTS_FILE);
        require(is_regular_file(&path), "Shared weights archive is missing")?;
        let expected_weight_bytes = contract_obj["bytes"].as_u64().unwrap();
        let expected_weight_sha = contract_obj["sha256"].as_str().unwrap();
        require(
            fs::metadata(&path)?.len() == expected_weight_bytes
                && sha256_file(&path)? == expected_weight_sha,
            "Shared weights archive checksum mismatch",
        )?;

        let (file_size, safetensors_metadata, manifest) =
            parse_embedded_manifest(&path).map_err(|err| {
                Error::new(format!("Shared weights archive metadata is invalid: {err}"))
            })?;

        require(
            Some(manifest.identity_sha256.as_str())
                == contract_obj.get("identity").and_then(Value::as_str),
            "Shared weights identity mismatch",
        )?;

        require(
            manifest.entries.len() == 2
                && manifest.entries.contains_key("prefill")
                && manifest.entries.contains_key("decode"),
            "Shared weights manifest requires both package entries",
        )?;

        for kind in ["prefill", "decode"] {
            let entry = &manifest.entries[kind];
            require(
                package_digests[kind] == entry.package_sha256,
                format!("Shared weights package checksum mismatch: {kind}"),
            )?;
            for target in entry.constants.values() {
                require(
                    manifest.tensors.contains_key(target),
                    "Shared weights constant mapping is incomplete",
                )?;
            }
        }

        Ok(Self {
            directory: root,
            contract: contract.clone(),
            manifest,
            safetensors_metadata,
            file_size,
        })
    }

    /// Returns the logical model identity object embedded in the weight archive.
    pub fn identity(&self) -> &Value {
        &self.manifest.identity
    }
}

fn parse_embedded_manifest(path: &Path) -> Result<(u64, SafetensorsMetadata, EmbeddedManifest)> {
    let (file_size, st_meta) = read_safetensors_metadata(path)?;
    let custom_meta = st_meta
        .metadata()
        .as_ref()
        .ok_or_else(|| Error::new("Shared weights archive metadata is missing"))?;
    let raw_manifest = custom_meta
        .get(SHARED_WEIGHTS_MANIFEST_KEY)
        .ok_or_else(|| Error::new("Shared weights archive metadata is missing"))?;
    let manifest: EmbeddedManifest = serde_json::from_str(raw_manifest)?;
    Ok((file_size, st_meta, manifest))
}

/// Verifies a strided view `(shape, stride, offset)` stays within `[0, numel)` without `i64` overflow.
pub fn check_view_bounds(shape: &[i64], stride: &[i64], offset: i64, numel: i64) -> Result<()> {
    require(
        shape.len() == stride.len()
            && offset >= 0
            && shape.iter().all(|&n| n >= 0)
            && stride.iter().all(|&n| n >= 0),
        "Shared weights tensor layout is invalid",
    )?;
    if shape.contains(&0) {
        require(offset <= numel, "Shared weights tensor layout is invalid")?;
        return Ok(());
    }

    let mut max_index = offset;
    for (&dim, &step) in shape.iter().zip(stride.iter()) {
        let span = dim - 1;
        if span == 0 || step == 0 {
            continue;
        }
        require(
            step <= (i64::MAX - max_index) / span,
            "Shared weights tensor layout is invalid",
        )?;
        max_index += span * step;
    }
    require(max_index < numel, "Shared weights tensor layout is invalid")
}

/// Canonical owner of all weight storage buffers and their strided parameter views.
/// Must outlive every `Package` bound via `bind_to_package` (`user_managed = true`).
pub struct SharedWeights {
    bundle: ValidatedBundle,
    _storages: HashMap<String, Tensor>,
    tensors: HashMap<String, Tensor>,
}

impl SharedWeights {
    pub fn load(bundle: ValidatedBundle, contract: &Value) -> Result<Self> {
        require(
            &bundle.contract == contract,
            "Shared weights contracts differ",
        )?;

        let archive_tensors = bundle.safetensors_metadata.tensors();
        require(
            archive_tensors.len() == bundle.manifest.storages.len(),
            "Shared weights archive buffer set mismatch",
        )?;

        struct StorageEntry {
            dtype: DType,
            cuda: bool,
            numel: i64,
        }

        let mut entries = HashMap::with_capacity(bundle.manifest.storages.len());

        // Validate every storage entry and tensor view before mapping or allocating device memory.
        for (name, descriptor) in &bundle.manifest.storages {
            let info = archive_tensors
                .get(name)
                .ok_or_else(|| Error::new("Shared weights archive buffer set mismatch"))?;
            require(
                descriptor.device == "cpu" || descriptor.device == "cuda",
                "Shared weights constant has unsupported device",
            )?;
            let dtype = DType::from_manifest_name(&descriptor.dtype)?;

            require(
                info.dtype == dtype.to_safetensors() && info.shape.len() == 1,
                "Shared weights archive tensor metadata mismatch",
            )?;
            let numel = i64::try_from(info.shape[0])
                .map_err(|_| Error::new("Shared weights archive tensor metadata mismatch"))?;
            let elem_size = dtype.element_size() as u64;
            let (begin, end) = info.data_offsets;
            let actual_bytes = (end - begin) as u64;
            require(
                descriptor.bytes <= i64::MAX as u64
                    && (numel as u64) <= u64::MAX / elem_size
                    && (numel as u64) * elem_size == descriptor.bytes
                    && actual_bytes == descriptor.bytes,
                "Shared weights archive tensor metadata mismatch",
            )?;

            entries.insert(
                name.clone(),
                StorageEntry {
                    dtype,
                    cuda: descriptor.device == "cuda",
                    numel,
                },
            );
        }

        for descriptor in bundle.manifest.tensors.values() {
            let storage_entry = entries
                .get(&descriptor.storage)
                .ok_or_else(|| Error::new("Shared weights tensor storage metadata mismatch"))?;
            let storage_desc = &bundle.manifest.storages[&descriptor.storage];
            require(
                descriptor.dtype == storage_desc.dtype && descriptor.device == storage_desc.device,
                "Shared weights tensor storage metadata mismatch",
            )?;
            check_view_bounds(
                &descriptor.shape,
                &descriptor.stride,
                descriptor.offset,
                storage_entry.numel,
            )?;
        }
        drop(archive_tensors);

        let need_map = !entries.is_empty() && bundle.safetensors_metadata.data_len() > 0;
        let mapped = MappedFile::open(
            &bundle.directory.join(SHARED_WEIGHTS_FILE),
            if need_map { bundle.file_size } else { 0 },
        )?;
        let st = if need_map {
            Some(
                SafeTensors::deserialize(mapped.as_slice())
                    .map_err(|err| Error::new(format!("Invalid safetensors archive: {err}")))?,
            )
        } else {
            None
        };

        let mut storages = HashMap::with_capacity(entries.len());
        let mut copied_cuda = false;
        for (name, entry) in entries {
            let device = if entry.cuda {
                Device::Cuda(0)
            } else {
                Device::Cpu
            };
            if entry.numel == 0 {
                storages.insert(name, Tensor::empty_contiguous(&[0], entry.dtype, device)?);
                continue;
            }

            let view = st
                .as_ref()
                .ok_or_else(|| Error::new("Shared weights archive is not mapped"))?
                .tensor(&name)
                .map_err(|err| Error::new(format!("Missing storage tensor {name}: {err}")))?;
            let slice = view.data();

            let mut dst = Tensor::empty_contiguous(&[entry.numel], entry.dtype, device)?;
            if entry.cuda {
                dst.copy_bytes_h2d_async(slice)?;
                copied_cuda = true;
            } else {
                dst.copy_bytes_cpu(slice)?;
            }
            storages.insert(name, dst);
        }

        // Complete all H2D copies before dropping the SafeTensors view and unmapping backing pages.
        if copied_cuda {
            cuda_synchronize()?;
        }
        drop(st);
        drop(mapped);

        let mut tensors = HashMap::with_capacity(bundle.manifest.tensors.len());
        for (name, descriptor) in &bundle.manifest.tensors {
            let storage = &storages[&descriptor.storage];
            let tensor =
                storage.reinterpret(&descriptor.shape, &descriptor.stride, descriptor.offset)?;
            tensors.insert(name.clone(), tensor);
        }

        Ok(Self {
            bundle,
            _storages: storages,
            tensors,
        })
    }

    pub fn package_path(&self, kind: &str) -> Result<PathBuf> {
        require(
            kind == "prefill" || kind == "decode",
            format!("Unknown package entry: {kind}"),
        )?;
        Ok(self.bundle.directory.join(format!("{kind}.pt2")))
    }

    /// Verifies exact FQN parity between `pkg` and the embedded manifest, then binds
    /// retained weight views into `pkg` without copying.
    pub fn bind_to_package(&self, pkg: &mut Package, kind: &str) -> Result<()> {
        let mapping = &self.bundle.manifest.entries[kind].constants;
        let names = pkg.constant_fqns()?;
        let unique_names: HashSet<&str> = names.iter().map(String::as_str).collect();
        require(
            names.len() == mapping.len() && unique_names.len() == names.len(),
            format!("Shared weights constant FQNs differ: {kind}"),
        )?;

        let mut pairs = Vec::with_capacity(names.len());
        for name in &names {
            let tensor_key = mapping.get(name).ok_or_else(|| {
                Error::new(format!("Shared weights constant FQNs differ: {kind}"))
            })?;
            let tensor = self
                .tensors
                .get(tensor_key)
                .ok_or_else(|| Error::new("Shared weights constant mapping is incomplete"))?;
            pairs.push((name.as_str(), tensor));
        }

        pkg.load_constants(&pairs)
    }

    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }
}
