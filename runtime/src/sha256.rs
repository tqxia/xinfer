//! Streaming SHA-256 file digest for bundle integrity verification.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

const BUFFER_BYTES: usize = 256 * 1024;

/// Streaming SHA-256 with automatic hardware acceleration (x86_64 SHA-NI + ARMv8 SHA2).
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|_| {
        Error::new(format!(
            "Shared weights cannot open file for checksum: {}",
            path.display()
        ))
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; BUFFER_BYTES];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|_| Error::new("Shared weights checksum read failed"))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}
