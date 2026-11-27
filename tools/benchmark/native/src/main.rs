//! Fixed-token native benchmark worker (`xinfer_benchmark`).
//!
//! Measures CUDA initialization, package load, reset, prefill, and fixed-length
//! decode timings plus allocator/device memory watermarks, verifying repeated
//! prediction determinism across warmup and measured samples.

use std::env;
use std::ffi::c_int;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use xinfer_runtime::{
    checked_device, create_output_file, cuda_mem_info, cuda_synchronize, path_exists, require,
    torch_build_info, DType, Device, Error, Model, Result, Session, Spec, Tensor,
};

#[repr(C)]
struct XinferCudaAllocStats {
    allocated_bytes: u64,
    reserved_bytes: u64,
    peak_allocated_bytes: u64,
    peak_reserved_bytes: u64,
}

extern "C" {
    fn xinfer_bench_cuda_alloc_stats(out: *mut XinferCudaAllocStats);
    fn xinfer_bench_cuda_reset_peak_stats();
    fn xinfer_bench_get_num_threads() -> c_int;
}

#[derive(Deserialize)]
struct Workload {
    warmup: i64,
    repeats: i64,
    cases: Vec<WorkloadCase>,
    #[serde(default)]
    max_prompt_length: Option<Value>,
}

#[derive(Deserialize)]
struct WorkloadCase {
    name: String,
    prompt: Vec<i64>,
    decode_tokens: Vec<i64>,
}

#[derive(Serialize)]
struct MemoryStats {
    allocated_bytes: u64,
    reserved_bytes: u64,
    peak_allocated_bytes: u64,
    peak_reserved_bytes: u64,
    device_used_bytes: usize,
    device_total_bytes: usize,
}

#[derive(Serialize)]
struct CaseMemoryStats {
    #[serde(flatten)]
    base: MemoryStats,
    sampled_peak_device_used_bytes: usize,
}

#[derive(Serialize)]
struct RequestMetrics {
    reset_ms: f64,
    prefill_ms: f64,
    first_prediction_ms: f64,
    decode_ms: f64,
    decode_ms_per_token: f64,
    request_ms: f64,
    prediction_ids: Vec<i64>,
    consumed_positions: i64,
}

#[derive(Serialize)]
struct CaseReport {
    name: String,
    prompt_tokens: usize,
    decode_steps: usize,
    first_request: RequestMetrics,
    samples: Vec<RequestMetrics>,
    memory: CaseMemoryStats,
}

#[derive(Serialize)]
struct BenchmarkReport {
    profile: &'static str,
    torch: String,
    cpu_threads: i32,
    cuda_init_ms: f64,
    load_ms: f64,
    initial_memory: MemoryStats,
    load_memory: MemoryStats,
    cases: Vec<CaseReport>,
}

fn cuda_alloc_stats() -> XinferCudaAllocStats {
    let mut stats = XinferCudaAllocStats {
        allocated_bytes: 0,
        reserved_bytes: 0,
        peak_allocated_bytes: 0,
        peak_reserved_bytes: 0,
    };
    unsafe {
        xinfer_bench_cuda_alloc_stats(&mut stats);
    }
    stats
}

fn reset_peak_stats() {
    unsafe {
        xinfer_bench_cuda_reset_peak_stats();
    }
}

fn num_threads() -> i32 {
    unsafe { xinfer_bench_get_num_threads() }
}

fn milliseconds(start: Instant, end: Instant) -> f64 {
    end.duration_since(start).as_secs_f64() * 1000.0
}

/// Snapshots both PyTorch's caching allocator stats and `cudaMemGetInfo` device usage.
fn memory_stats() -> Result<MemoryStats> {
    let (free, total) = cuda_mem_info()?;
    let alloc = cuda_alloc_stats();
    Ok(MemoryStats {
        allocated_bytes: alloc.allocated_bytes,
        reserved_bytes: alloc.reserved_bytes,
        peak_allocated_bytes: alloc.peak_allocated_bytes,
        peak_reserved_bytes: alloc.peak_reserved_bytes,
        device_used_bytes: total - free,
        device_total_bytes: total,
    })
}

struct Options {
    packages: PathBuf,
    workload: PathBuf,
    output: PathBuf,
}

fn parse_options(args: &[String]) -> Result<Options> {
    require(
        args.len() == 4,
        "Usage: xinfer_benchmark PACKAGE_DIR WORKLOAD_JSON OUTPUT_JSON",
    )?;
    Ok(Options {
        packages: PathBuf::from(&args[1]),
        workload: PathBuf::from(&args[2]),
        output: PathBuf::from(&args[3]),
    })
}

fn read_workload(path: &Path) -> Result<Workload> {
    let content = fs::read_to_string(path)
        .map_err(|_| Error::new(format!("Cannot open metadata: {}", path.display())))?;
    let workload: Workload = serde_json::from_str(&content)?;
    Ok(workload)
}

fn validate_workload(workload: &Workload, spec: &Spec) -> Result<()> {
    require(
        workload.warmup > 0 && workload.repeats > 0 && !workload.cases.is_empty(),
        "Invalid benchmark counts",
    )?;

    for item in &workload.cases {
        spec.check_request(&item.prompt, item.decode_tokens.len() as i64)?;
        for &id in &item.decode_tokens {
            require(id >= 0 && id < spec.vocab, "Decode token ID out of range")?;
        }
    }
    Ok(())
}

/// Runs one synchronized `reset -> prefill -> fixed decode` request and verifies
/// all supplied tokens (including EOS) were consumed.
fn measure_request(
    session: &mut Session,
    prompt: &[i64],
    tokens: &[i64],
) -> Result<RequestMetrics> {
    let mut predictions = Vec::with_capacity(tokens.len() + 1);

    cuda_synchronize()?;
    let begin = Instant::now();
    session.reset(0.0)?;
    cuda_synchronize()?;

    let prefill = Instant::now();
    predictions.push(session.prefill(prompt)?.argmax_last_dim()?);
    cuda_synchronize()?;

    let decode = Instant::now();
    // EOS remains data: every profile consumes the identical fixed workload.
    for &token in tokens {
        predictions.push(session.step(&[token])?.argmax_last_dim()?);
    }
    cuda_synchronize()?;
    let end = Instant::now();

    require(
        session.position() == (prompt.len() + tokens.len()) as i64,
        "Consumed position mismatch",
    )?;

    let decode_ms = milliseconds(decode, end);
    Ok(RequestMetrics {
        reset_ms: milliseconds(begin, prefill),
        prefill_ms: milliseconds(prefill, decode),
        first_prediction_ms: milliseconds(begin, decode),
        decode_ms,
        decode_ms_per_token: decode_ms / (tokens.len() as f64),
        request_ms: milliseconds(begin, end),
        prediction_ids: predictions,
        consumed_positions: session.position(),
    })
}

fn measure_cases(session: &mut Session, workload: &Workload) -> Result<Vec<CaseReport>> {
    let warmup = workload.warmup as usize;
    let repeats = workload.repeats as usize;
    let mut cases = Vec::with_capacity(workload.cases.len());

    for item in &workload.cases {
        let first = measure_request(session, &item.prompt, &item.decode_tokens)?;
        let mut samples = Vec::with_capacity(repeats);
        let mut peak_device = memory_stats()?.device_used_bytes;

        for i in 0..(warmup + repeats) {
            if i == warmup {
                reset_peak_stats();
            }
            let sample = measure_request(session, &item.prompt, &item.decode_tokens)?;
            require(
                sample.prediction_ids == first.prediction_ids,
                "Repeated predictions differ",
            )?;
            peak_device = peak_device.max(memory_stats()?.device_used_bytes);
            if i >= warmup {
                samples.push(sample);
            }
        }

        let memory = CaseMemoryStats {
            base: memory_stats()?,
            sampled_peak_device_used_bytes: peak_device,
        };
        cases.push(CaseReport {
            name: item.name.clone(),
            prompt_tokens: item.prompt.len(),
            decode_steps: item.decode_tokens.len(),
            first_request: first,
            samples,
            memory,
        });
        eprintln!("{}: completed", item.name);
    }

    Ok(cases)
}

fn write_report(path: &Path, report: &BenchmarkReport) -> Result<()> {
    let mut writer = BufWriter::new(create_output_file(path)?);
    let text = serde_json::to_string_pretty(report)?;
    writeln!(writer, "{text}")
        .and_then(|()| writer.flush())
        .map_err(|_| Error::new("Failed to write benchmark report"))
}

fn run_benchmark(options: &Options) -> Result<()> {
    let workload = read_workload(&options.workload)?;
    require(!path_exists(&options.output), "Output already exists")?;
    require(
        workload.max_prompt_length.is_none(),
        "Benchmark workload no longer accepts max_prompt_length",
    )?;

    let model = Model::open(&options.packages)?;
    validate_workload(&workload, model.spec())?;

    let init = Instant::now();
    checked_device(model.spec())?;
    let _empty = Tensor::empty_contiguous(&[0], DType::Float32, Device::Cuda(0))?;
    cuda_synchronize()?;
    let init_ms = milliseconds(init, Instant::now());
    let initial_memory = memory_stats()?;
    reset_peak_stats();

    let load = Instant::now();
    let mut session = model.open_session(Some(1))?;
    cuda_synchronize()?;
    let load_ms = milliseconds(load, Instant::now());

    // Retain `profile: "cpp_aoti"` for compatibility with Python benchmark aggregators.
    let report = BenchmarkReport {
        profile: "cpp_aoti",
        torch: torch_build_info().torch_version,
        cpu_threads: num_threads(),
        cuda_init_ms: init_ms,
        load_ms,
        initial_memory,
        load_memory: memory_stats()?,
        cases: measure_cases(&mut session, &workload)?,
    };

    write_report(&options.output, &report)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match parse_options(&args).and_then(|opts| run_benchmark(&opts)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires GPU and XINFER_PACKAGES"]
    fn session_open_close_has_zero_vram_leak_and_sets_single_thread() {
        let dir = PathBuf::from(
            env::var("XINFER_PACKAGES").expect("Set XINFER_PACKAGES to a compiled 4-file bundle"),
        );
        let model = Model::open(&dir).unwrap();

        // First cycle warms up process-global cuBLAS workspaces (~8 MiB persistent).
        {
            let mut session = model.open_session(Some(1)).unwrap();
            assert_eq!(num_threads(), 1, "Runtime must configure 1 CPU thread");
            let _ = session.generate(&[42, 13], 3, None).unwrap();
        }
        cuda_synchronize().unwrap();
        let after_first = cuda_alloc_stats().allocated_bytes;

        // Second open/run/drop cycle must return to the exact same allocated_bytes (0 bytes leaked).
        let while_open;
        {
            let mut session = model.open_session(Some(1)).unwrap();
            let _ = session.generate(&[42, 13], 3, None).unwrap();
            cuda_synchronize().unwrap();
            while_open = cuda_alloc_stats().allocated_bytes;
            assert!(while_open > after_first);
        }
        cuda_synchronize().unwrap();
        let after_second = cuda_alloc_stats().allocated_bytes;
        eprintln!(
            "Exact CUDA allocated_bytes: after_first={after_first} while_open={while_open} after_second={after_second}"
        );
        assert_eq!(
            after_second, after_first,
            "VRAM leaked across Session open/drop cycle"
        );
    }
}
