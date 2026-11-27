//! Deterministic per-request RNG and FFI wrapper over `xinfer_shim_sample_batch`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use xinfer_runtime::torch_sys::{self, shim_call, XinferSampleRow};
use xinfer_runtime::{require_kind, ErrorKind, Result, Tensor};

use crate::types::SamplingParams;

/// Per-row sampling inputs passed into `BatchBackend::sample_batch`.
#[derive(Clone, Copy, Debug)]
pub struct SampleRow {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub uniform: f32,
}

impl SampleRow {
    pub fn from_params(params: &SamplingParams, rng: &mut RequestRng) -> Self {
        let uniform = if params.temperature > 0.0 {
            rng.next_uniform_f32()
        } else {
            0.0
        };
        Self {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            uniform,
        }
    }
}

/// Lightweight, reproducible `Xoshiro256**` PRNG seeded via `SplitMix64`.
#[derive(Clone, Debug)]
pub struct RequestRng {
    state: [u64; 4],
}

static AUTO_SEED_COUNTER: AtomicU64 = AtomicU64::new(1);

impl RequestRng {
    pub fn from_seed(seed: Option<u64>) -> Self {
        let mut z = seed.unwrap_or_else(|| {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            let seq = AUTO_SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
            nanos ^ seq.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        });
        let mut next_splitmix = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            state: [
                next_splitmix(),
                next_splitmix(),
                next_splitmix(),
                next_splitmix(),
            ],
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Returns a uniform float in `[0.0, 1.0)` with 24 bits of mantissa precision.
    pub fn next_uniform_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0f32 / ((1u32 << 24) as f32))
    }
}

/// Samples one token ID per active row of 2D `[batch, vocab]` logits on the logits' resident device.
pub fn sample_batch(logits: &Tensor, rows: &[SampleRow], active: &[bool]) -> Result<Vec<i64>> {
    require_kind(
        rows.len() == active.len(),
        ErrorKind::InvalidInput,
        "Sampler rows/active length mismatch",
    )?;
    let count = rows.len();
    if count == 0 {
        return Ok(Vec::new());
    }
    let c_params: Vec<XinferSampleRow> = rows
        .iter()
        .map(|r| XinferSampleRow {
            temperature: r.temperature,
            top_p: r.top_p,
            top_k: r.top_k,
            uniform: r.uniform,
        })
        .collect();
    let c_active: Vec<u8> = active.iter().map(|&flag| u8::from(flag)).collect();
    let mut out_tokens = vec![0i64; count];
    shim_call(|err| unsafe {
        torch_sys::xinfer_shim_sample_batch(
            logits.raw(),
            c_params.as_ptr(),
            c_active.as_ptr(),
            out_tokens.as_mut_ptr(),
            count,
            err,
        )
    })?;
    Ok(out_tokens)
}
