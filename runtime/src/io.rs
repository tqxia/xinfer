//! Safe filesystem path checks, no-overwrite output creation, and raw tensor I/O.

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Detect existing paths including dangling symlinks (matching C++ `fs::exists(fs::symlink_status)`).
pub fn path_exists(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok()
}

/// Lexical + canonical path resolution matching C++ `std::filesystem::weakly_canonical`.
pub fn weakly_canonical(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    };

    let mut resolved = PathBuf::new();
    let mut still_exists = true;
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => resolved.push(prefix.as_os_str()),
            Component::RootDir => resolved.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if still_exists {
                    let candidate = resolved.join("..");
                    if let Ok(canon) = fs::canonicalize(&candidate) {
                        resolved = canon;
                    } else {
                        still_exists = false;
                        resolved.pop();
                    }
                } else {
                    resolved.pop();
                }
            }
            Component::Normal(segment) => {
                resolved.push(segment);
                if still_exists {
                    if let Ok(canon) = fs::canonicalize(&resolved) {
                        resolved = canon;
                    } else {
                        still_exists = false;
                    }
                }
            }
        }
    }
    Ok(resolved)
}

/// Atomically creates a new output file (`O_CREAT | O_EXCL`), failing if it exists.
pub fn create_output_file(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|_| Error::new(format!("Cannot open output: {}", path.display())))?;
        }
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|_| Error::new(format!("Cannot open output: {}", path.display())))
}

/// Creates or truncates an output file (used for incremental report updates after
/// preflight has already confirmed the destination was initially absent).
pub fn write_output_file(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|_| Error::new(format!("Cannot open output: {}", path.display())))?;
        }
    }
    File::create(path).map_err(|_| Error::new(format!("Cannot open output: {}", path.display())))
}

/// Writes raw contiguous element bytes of `tensor` in host endianness.
pub fn write_tensor_bytes(writer: &mut dyn Write, tensor: &Tensor) -> Result<()> {
    let bytes = tensor.to_contiguous_host_bytes()?;
    writer
        .write_all(&bytes)
        .map_err(|_| Error::new("Failed to write tensor"))
}
