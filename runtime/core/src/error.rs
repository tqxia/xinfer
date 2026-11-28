//! Lightweight classified runtime error type and precondition helpers.

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

/// Coarse failure category used by higher-level layers (e.g., HTTP status mapping)
/// while keeping CLI stderr messages unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidInput,
    CapacityExceeded,
    QueueFull,
    Unavailable,
    Unsupported,
    Io,
    Runtime,
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Runtime,
            message: message.into(),
        }
    }

    pub fn with_kind(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::with_kind(ErrorKind::InvalidInput, message)
    }

    pub fn capacity_exceeded(message: impl Into<String>) -> Self {
        Self::with_kind(ErrorKind::CapacityExceeded, message)
    }

    pub fn queue_full(message: impl Into<String>) -> Self {
        Self::with_kind(ErrorKind::QueueFull, message)
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::with_kind(ErrorKind::Unavailable, message)
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::with_kind(ErrorKind::Unsupported, message)
    }

    pub fn io(message: impl Into<String>) -> Self {
        Self::with_kind(ErrorKind::Io, message)
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::io(value.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::invalid_input(value.to_string())
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(value: std::ffi::NulError) -> Self {
        Self::invalid_input(value.to_string())
    }
}

/// Fails with `message` when `condition` is false (mirrors C++ `xinfer::require`).
pub fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::new(message))
    }
}

/// Fails with a classified `ErrorKind` when `condition` is false.
pub fn require_kind(condition: bool, kind: ErrorKind, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::with_kind(kind, message))
    }
}
