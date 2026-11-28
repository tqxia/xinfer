//! Native validation worker (`xinfer_validate`) driven by Python validation suites.
//!
//! Executes trace (single-row or dynamic `batch_steps`) and greedy cases against a
//! compiled bundle, enforces per-step write-region / alignment / reset / overflow
//! invariants, verifies `/proc/self/maps` has no Python linkage, and optionally
//! checks bitwise trace parity against a reference directory.

use std::collections::HashSet;
use std::env;
use std::ffi::{c_int, OsStr};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::{Deserialize, Serialize};

use xinfer_runtime::torch_sys::{shim_call, AtenTensorHandle, XinferError};
use xinfer_runtime::{
    create_output_file, path_exists, require, write_logits_f32, write_output_file,
    write_tensor_bytes, Error, Model, ModelType, Result, Session, Spec, StateSpec, Tensor,
};

extern "C" {
    fn xinfer_validate_all_finite(
        tensor: AtenTensorHandle,
        out: *mut bool,
        error: *mut XinferError,
    ) -> c_int;
    fn xinfer_validate_equal(
        a: AtenTensorHandle,
        b: AtenTensorHandle,
        out: *mut bool,
        error: *mut XinferError,
    ) -> c_int;
    fn xinfer_validate_all_eq_scalar(
        tensor: AtenTensorHandle,
        value: f64,
        out: *mut bool,
        error: *mut XinferError,
    ) -> c_int;
    fn xinfer_validate_all_eq_fill_bytes(
        tensor: AtenTensorHandle,
        fill: f64,
        out: *mut bool,
        error: *mut XinferError,
    ) -> c_int;
}

fn default_trace_format() -> i64 {
    1
}

#[derive(Deserialize)]
struct CaseManifest {
    #[serde(default = "default_trace_format")]
    trace_format: i64,
    #[serde(default)]
    expected_directory: String,
    cases: Vec<RawCase>,
}

#[derive(Deserialize)]
struct RawCase {
    name: String,
    mode: Option<String>,
    prompt: Option<Vec<i64>>,
    limit: Option<i64>,
    eos: Option<Vec<i64>>,
    #[serde(default)]
    dump_cache: bool,
    #[serde(default)]
    check_overflow: bool,
    fill: Option<f64>,
    poison_kv: Option<f64>,
    steps: Option<Vec<Vec<i64>>>,
    chunks: Option<Vec<Vec<i64>>>,
    batch_steps: Option<Vec<Vec<Vec<i64>>>>,
    #[serde(default)]
    state_steps: Vec<i64>,
}

#[derive(Default, Serialize)]
struct CaseReport {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    state_steps: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    state_prefix: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ids: Option<Vec<i64>>,
    rows: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    steps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    write_region_checks: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reset_check: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    state_inputs_updated_in_place: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_write_regions_preserved: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    overflow_rejected: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    overflow_rejected_without_mutation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bitwise_parity: Option<bool>,
    position: i64,
    positions: Vec<i64>,
    stable_storage: bool,
    stable_kv_storage: bool,
    prefill_calls: i64,
    decode_calls: i64,
}

#[derive(Serialize)]
struct ValidationReport {
    cases: Vec<CaseReport>,
    loader_pairs: usize,
    state_tensors: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_buffers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    no_python_library: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attention_library: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    deltanet_library: Option<bool>,
}

fn all_finite(tensor: &Tensor) -> Result<bool> {
    let mut out = false;
    shim_call(|err| unsafe { xinfer_validate_all_finite(tensor.raw(), &mut out, err) })?;
    Ok(out)
}

fn tensors_equal(a: &Tensor, b: &Tensor) -> Result<bool> {
    let mut out = false;
    shim_call(|err| unsafe { xinfer_validate_equal(a.raw(), b.raw(), &mut out, err) })?;
    Ok(out)
}

fn all_eq_scalar(tensor: &Tensor, value: f64) -> Result<bool> {
    let mut out = false;
    shim_call(|err| unsafe { xinfer_validate_all_eq_scalar(tensor.raw(), value, &mut out, err) })?;
    Ok(out)
}

fn all_eq_fill_bytes(tensor: &Tensor, fill: f64) -> Result<bool> {
    let mut out = false;
    shim_call(|err| unsafe {
        xinfer_validate_all_eq_fill_bytes(tensor.raw(), fill, &mut out, err)
    })?;
    Ok(out)
}

struct Options {
    packages: PathBuf,
    cases: PathBuf,
    output: PathBuf,
}

fn parse_options(args: &[String]) -> Result<Options> {
    require(
        args.len() == 4,
        "Usage: xinfer_validate PACKAGE_DIR CASES_JSON OUTPUT_DIR",
    )?;
    Ok(Options {
        packages: PathBuf::from(&args[1]),
        cases: PathBuf::from(&args[2]),
        output: PathBuf::from(&args[3]),
    })
}

/// Controls trace file layout (`trace_format` 1 vs 2 and Qwen3 vs Qwen3.5/3.8 conventions):
/// - `separate_logits`: write FP32 logits to `<case>.f32` instead of interleaved `.states`.
/// - `selected_states`: dump state only at `state_steps` (`<case>.<step>.states`).
/// - `compact`: for KV tensors, dump `[0..position)` prefix and verify `[position..cap)` equals `fill`.
#[derive(Clone, Copy)]
struct TraceLayout {
    separate_logits: bool,
    selected_states: bool,
    compact: bool,
}

struct Case {
    name: String,
    greedy: bool,
    dump_cache: bool,
    check_overflow: bool,
    prompt: Vec<i64>,
    eos: Vec<i64>,
    steps: Vec<Vec<i64>>,
    batch_steps: Vec<Vec<Vec<i64>>>,
    limit: i64,
    fill: f64,
    override_eos: bool,
    state_steps: Vec<i64>,
}

fn read_manifest(path: &Path) -> Result<CaseManifest> {
    let content = fs::read_to_string(path)
        .map_err(|_| Error::new(format!("Cannot open case manifest: {}", path.display())))?;
    let manifest: CaseManifest = serde_json::from_str(&content)?;
    Ok(manifest)
}

fn parse_cases(manifest: CaseManifest, spec: &Spec, output: &Path) -> Result<Vec<Case>> {
    require(
        !path_exists(output) || output.is_dir(),
        "Output must be a directory",
    )?;
    require(
        !path_exists(&output.join("native.json")),
        "Output already exists",
    )?;

    let trace_format = manifest.trace_format;
    let expected_dir = manifest.expected_directory;
    let mut names = HashSet::new();
    let mut cases = Vec::with_capacity(manifest.cases.len());

    for raw in manifest.cases {
        require(
            !raw.name.is_empty()
                && raw.name != "."
                && raw.name != ".."
                && !raw.name.contains('/')
                && Path::new(&raw.name).file_name() == Some(OsStr::new(&raw.name))
                && names.insert(raw.name.clone()),
            "Invalid or duplicate case name",
        )?;

        let greedy = raw.prompt.is_some();
        if let Some(mode) = raw.mode.as_deref() {
            require(
                mode == "greedy" || mode == "trace",
                "Unknown validation mode",
            )?;
            require(
                (mode == "greedy") == greedy,
                "Validation mode disagrees with inputs",
            )?;
        }

        let fill = raw.fill.or(raw.poison_kv).unwrap_or(0.0);
        let override_eos = raw.eos.is_some();
        let eos = raw.eos.unwrap_or_else(|| spec.eos.clone());
        require(!eos.is_empty(), "Missing EOS tokens")?;
        for &token in &eos {
            require(
                token >= 0 && token < spec.vocab,
                "EOS token ID out of range",
            )?;
        }

        let mut prompt = Vec::new();
        let mut limit = 0i64;
        let mut steps = Vec::new();
        let mut batch_steps = Vec::new();
        let mut state_steps = Vec::new();

        if greedy {
            prompt = raw.prompt.unwrap();
            limit = raw.limit.ok_or_else(|| Error::new("Missing limit"))?;
            spec.check_request(&prompt, limit)?;
        } else {
            if let Some(raw_batch) = raw.batch_steps {
                require(
                    raw.chunks.is_none() && raw.steps.is_none(),
                    "Expected one validation input form",
                )?;
                batch_steps = raw_batch;
                for rows in &batch_steps {
                    let slices: Vec<&[i64]> = rows.iter().map(Vec::as_slice).collect();
                    spec.check_batch_tokens(&slices)?;
                    steps.push(rows[0].clone());
                }
            } else {
                require(
                    raw.chunks.is_some() != raw.steps.is_some(),
                    "Expected validation chunks or steps",
                )?;
                steps = raw.steps.or(raw.chunks).unwrap();
            }
            require(!steps.is_empty(), "Expected validation steps")?;

            let mut positions = vec![0i64; spec.max_batch_size as usize];
            for step_idx in 0..steps.len() {
                let row_slices: Vec<&[i64]> = if batch_steps.is_empty() {
                    vec![steps[step_idx].as_slice()]
                } else {
                    batch_steps[step_idx].iter().map(Vec::as_slice).collect()
                };
                spec.check_batch_tokens(&row_slices)?;
                for (row_idx, row) in row_slices.iter().enumerate() {
                    let len = row.len() as i64;
                    require(
                        len <= spec.capacity - positions[row_idx],
                        "Context capacity exceeded",
                    )?;
                    positions[row_idx] += len;
                    require(positions[row_idx] > 0, "Each row must start with tokens")?;
                }
            }
            let max_pos = positions.iter().copied().max().unwrap_or(0);
            require(
                !raw.check_overflow || max_pos == spec.capacity,
                "Overflow case must fill the context",
            )?;

            if trace_format == 2 {
                state_steps = raw.state_steps;
                let mut previous = -1i64;
                for &step in &state_steps {
                    require(
                        step > previous && step < steps.len() as i64,
                        "Invalid state checkpoint indices",
                    )?;
                    previous = step;
                }
            }
        }

        let qwen3 = spec.model_type == ModelType::Qwen3;
        let separate_logits = qwen3 || !expected_dir.is_empty();
        let mut suffixes = Vec::new();
        if separate_logits || greedy {
            suffixes.push(".f32");
        }
        if !separate_logits {
            suffixes.push(".states");
        }
        if qwen3 && raw.dump_cache {
            suffixes.push(".kv.bf16");
        }
        for suffix in suffixes {
            require(
                !path_exists(&output.join(format!("{}{suffix}", raw.name))),
                "Output already exists",
            )?;
        }
        for &step in &state_steps {
            require(
                !path_exists(&output.join(format!("{}.{step}.states", raw.name))),
                "Output already exists",
            )?;
        }

        cases.push(Case {
            name: raw.name,
            greedy,
            dump_cache: raw.dump_cache,
            check_overflow: raw.check_overflow,
            prompt,
            eos,
            steps,
            batch_steps,
            limit,
            fill,
            override_eos,
            state_steps,
        });
    }

    require(!cases.is_empty(), "Expected validation cases")?;
    Ok(cases)
}

fn snapshot(session: &Session) -> Result<Vec<Tensor>> {
    session.state().iter().map(Tensor::clone_tensor).collect()
}

fn write_state(writer: &mut dyn Write, session: &Session, compact: bool, fill: f64) -> Result<()> {
    let pos = session.position();
    for (spec, tensor) in session.spec().state.iter().zip(session.state().iter()) {
        require(all_finite(tensor)?, "Non-finite state")?;
        if compact && spec.is_kv() {
            let cap = tensor.sizes()?[2];
            let future = tensor.slice_dim(2, pos, cap)?;
            require(all_eq_fill_bytes(&future, fill)?, "Future KV overwritten")?;
            let prefix = tensor.slice_dim(2, 0, pos)?;
            write_tensor_bytes(writer, &prefix)?;
        } else {
            write_tensor_bytes(writer, tensor)?;
        }
    }
    Ok(())
}

/// Audits `/proc/self/maps` to ensure no Python libraries or unexpected `libxinfer_*.so`
/// objects were loaded, while required bundled custom ops (attention / DeltaNet) are present.
fn check_native_libraries(report: &mut ValidationReport, model_type: ModelType) -> Result<()> {
    let maps = fs::read_to_string("/proc/self/maps")
        .map_err(|_| Error::new("Cannot inspect process mappings"))?;

    let mut attention = false;
    let mut deltanet = false;
    let expect_deltanet = matches!(model_type, ModelType::Qwen35 | ModelType::Qwen38);

    for line in maps.lines() {
        require(
            !line.contains("libpython") && !line.contains("libtorch_python"),
            "Python library loaded",
        )?;
        let is_attention = line.contains("libxinfer_attention_ops.so");
        let is_deltanet = line.contains("libxinfer_deltanet_ops.so");
        let expected = is_attention || (expect_deltanet && is_deltanet);
        require(
            !line.contains("libxinfer_") || expected,
            "Unexpected custom-op library loaded",
        )?;
        attention |= is_attention;
        deltanet |= is_deltanet;
    }

    require(attention, "Native attention library was not loaded")?;
    require(
        !expect_deltanet || deltanet,
        "Native DeltaNet library was not loaded",
    )?;
    report.no_python_library = Some(true);
    report.attention_library = Some(true);
    if expect_deltanet {
        report.deltanet_library = Some(true);
    }
    Ok(())
}

fn check_persistent(session: &Session, persistent: &[usize]) -> Result<()> {
    for (tensor, &expected_ptr) in session.state().iter().zip(persistent.iter()) {
        require(
            tensor.data_ptr()? as usize == expected_ptr,
            "State buffer storage changed",
        )?;
    }
    Ok(())
}

fn check_reset(session: &Session, fill: f64) -> Result<()> {
    for (spec, tensor) in session.spec().state.iter().zip(session.state().iter()) {
        let expected = if spec.is_kv() { fill } else { 0.0 };
        require(
            all_eq_scalar(tensor, expected)?,
            "State reset left stale data",
        )?;
    }
    Ok(())
}

/// Verifies in-place state update invariants for a single batch row after one step:
/// pointer/stride stability, past/future KV preservation, active KV mutation (or
/// inactive non-KV immutability), and finite values.
fn check_state_update(
    spec: &StateSpec,
    input_ptr: usize,
    input_strides: &[i64],
    before: &Tensor,
    after: &Tensor,
    start: i64,
    end: i64,
) -> Result<()> {
    require(
        after.data_ptr()? as usize == input_ptr && after.strides()? == input_strides,
        "State output lost caller-owned storage",
    )?;

    if spec.is_kv() {
        let cap = before.sizes()?[2];
        require(
            tensors_equal(
                &before.slice_dim(2, 0, start)?,
                &after.slice_dim(2, 0, start)?,
            )?,
            "Past KV overwritten",
        )?;
        require(
            tensors_equal(
                &before.slice_dim(2, end, cap)?,
                &after.slice_dim(2, end, cap)?,
            )?,
            "Future KV overwritten",
        )?;
        if end > start {
            require(
                !tensors_equal(
                    &before.slice_dim(2, start, end)?,
                    &after.slice_dim(2, start, end)?,
                )?,
                "Cache update missing",
            )?;
        }
    } else if start == end {
        require(tensors_equal(after, before)?, "Inactive state changed")?;
    }

    require(all_finite(after)?, "Non-finite state")
}

/// Confirms stepping past `capacity` fails with `Context capacity exceeded` without
/// mutating positions, state pointers, or state values.
fn check_overflow(session: &mut Session) -> Result<()> {
    let capacity = session.spec().capacity;
    let max_batch = session.spec().max_batch_size as usize;
    require(session.position() == capacity, "Context not full")?;

    let input_ptrs: Vec<usize> = session
        .state()
        .iter()
        .map(|t| Ok(t.data_ptr()? as usize))
        .collect::<Result<_>>()?;
    let before = snapshot(session)?;
    let positions = session.positions().to_vec();

    let one = [1i64];
    let rows = vec![one.as_slice(); max_batch];
    let rejected = match session.step_batch(&rows) {
        Ok(_) => false,
        Err(err) => err.to_string() == "Context capacity exceeded",
    };

    require(
        rejected && session.position() == capacity && session.positions() == positions.as_slice(),
        "Overflow not rejected",
    )?;
    for ((&ptr, expected), actual) in input_ptrs
        .iter()
        .zip(before.iter())
        .zip(session.state().iter())
    {
        require(
            actual.data_ptr()? as usize == ptr && tensors_equal(expected, actual)?,
            "Overflow changed state",
        )?;
    }
    Ok(())
}

fn run_greedy_case(
    session: &mut Session,
    c: &Case,
    layout: TraceLayout,
    logits: &mut Option<BufWriter<File>>,
    states: &mut Option<BufWriter<File>>,
    row: &mut CaseReport,
) -> Result<()> {
    let ids = {
        let writer = logits.as_mut().map(|w| w as &mut dyn Write);
        if c.override_eos {
            session.generate_with_eos(&c.prompt, c.limit, &c.eos, writer)?
        } else {
            session.generate(&c.prompt, c.limit, writer)?
        }
    };
    let stop = if ids.last().map(|t| c.eos.contains(t)).unwrap_or(false) {
        "eos"
    } else {
        "limit"
    };
    row.rows = ids.len();
    row.ids = Some(ids);
    row.stop = Some(stop);

    if !layout.separate_logits {
        if let Some(writer) = states.as_mut() {
            write_state(writer, session, layout.compact, 0.0)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_trace_case(
    session: &mut Session,
    c: &Case,
    layout: TraceLayout,
    output: &Path,
    persistent: &[usize],
    logits: &mut Option<BufWriter<File>>,
    states: &mut Option<BufWriter<File>>,
    row: &mut CaseReport,
) -> Result<()> {
    session.reset(c.fill)?;
    check_reset(session, c.fill)?;

    for step in 0..c.steps.len() {
        let input_meta: Vec<(usize, Vec<i64>)> = session
            .state()
            .iter()
            .map(|t| Ok((t.data_ptr()? as usize, t.strides()?)))
            .collect::<Result<_>>()?;
        let before = snapshot(session)?;
        let starts = session.positions().to_vec();

        let score = if c.batch_steps.is_empty() {
            session.step(&c.steps[step])?
        } else {
            let slices: Vec<&[i64]> = c.batch_steps[step].iter().map(Vec::as_slice).collect();
            session.step_batch(&slices)?
        };
        require(all_finite(&score)?, "Non-finite logits")?;

        let active_rows: Vec<&[i64]> = if c.batch_steps.is_empty() {
            vec![c.steps[step].as_slice()]
        } else {
            c.batch_steps[step].iter().map(Vec::as_slice).collect()
        };

        for (r, tokens) in active_rows.iter().enumerate() {
            require(
                session.positions()[r] == starts[r] + tokens.len() as i64,
                "Incorrect position advance",
            )?;
        }
        for (r, &start_pos) in starts.iter().enumerate().skip(active_rows.len()) {
            require(
                session.positions()[r] == start_pos,
                "Unused row position changed",
            )?;
        }

        for (r, tokens) in active_rows.iter().enumerate() {
            if tokens.is_empty() {
                let row_logits = score.narrow_dim0(r as i64, 1)?;
                require(
                    all_eq_scalar(&row_logits, 0.0)?,
                    "Inactive logits must be zero",
                )?;
            }
        }

        if layout.separate_logits {
            if let Some(writer) = logits.as_mut() {
                write_logits_f32(writer, &score)?;
            }
        } else if let Some(writer) = states.as_mut() {
            write_tensor_bytes(writer, &score)?;
        }

        require(
            session.state().len() == session.spec().state.len(),
            "Incorrect state count",
        )?;
        for (i, ((input_ptr, input_strides), before_t)) in
            input_meta.iter().zip(before.iter()).enumerate()
        {
            let after_t = &session.state()[i];
            let state_spec = &session.spec().state[i];
            let row_stride_bytes = (input_strides[0] as usize) * state_spec.dtype.element_size();
            for (r, &start_pos) in starts.iter().enumerate() {
                let expected_row_ptr = input_ptr + r * row_stride_bytes;
                let before_row = before_t.narrow_dim0(r as i64, 1)?;
                let after_row = after_t.narrow_dim0(r as i64, 1)?;
                check_state_update(
                    state_spec,
                    expected_row_ptr,
                    input_strides,
                    &before_row,
                    &after_row,
                    start_pos,
                    session.positions()[r],
                )?;
            }
            if !layout.separate_logits && !layout.selected_states {
                if let Some(writer) = states.as_mut() {
                    write_tensor_bytes(writer, after_t)?;
                }
            }
        }
        check_persistent(session, persistent)?;

        if layout.selected_states && c.state_steps.contains(&(step as i64)) {
            let ckpt_path = output.join(format!("{}.{}.states", c.name, step));
            let mut ckpt = BufWriter::new(create_output_file(&ckpt_path)?);
            write_state(&mut ckpt, session, layout.compact, c.fill)?;
            ckpt.flush()
                .map_err(|_| Error::new("Failed to write state checkpoint"))?;
        }
    }

    row.rows = c.steps.len();
    row.steps = Some(c.steps.len());
    row.write_region_checks = Some(true);
    row.reset_check = Some(true);
    row.state_inputs_updated_in_place = Some(true);
    row.kv_write_regions_preserved = Some(true);

    if session.spec().model_type == ModelType::Qwen3 && c.dump_cache && !layout.selected_states {
        let cache_path = output.join(format!("{}.kv.bf16", c.name));
        let mut cache = BufWriter::new(create_output_file(&cache_path)?);
        for tensor in session.state() {
            write_tensor_bytes(&mut cache, tensor)?;
        }
        cache
            .flush()
            .map_err(|_| Error::new("Failed to write cache dump"))?;
    }
    Ok(())
}

fn run_case(
    session: &mut Session,
    c: &Case,
    layout: TraceLayout,
    output: &Path,
    persistent: &[usize],
) -> Result<CaseReport> {
    let mut row = CaseReport {
        name: c.name.clone(),
        state_steps: layout.selected_states.then(|| c.state_steps.clone()),
        state_prefix: layout.selected_states.then_some(layout.compact),
        ..CaseReport::default()
    };

    let mut logits = if layout.separate_logits || c.greedy {
        Some(BufWriter::new(create_output_file(
            &output.join(format!("{}.f32", c.name)),
        )?))
    } else {
        None
    };
    let mut states = if !layout.separate_logits {
        Some(BufWriter::new(create_output_file(
            &output.join(format!("{}.states", c.name)),
        )?))
    } else {
        None
    };

    if c.greedy {
        run_greedy_case(session, c, layout, &mut logits, &mut states, &mut row)?;
    } else {
        run_trace_case(
            session,
            c,
            layout,
            output,
            persistent,
            &mut logits,
            &mut states,
            &mut row,
        )?;
    }

    if c.check_overflow {
        check_overflow(session)?;
        row.overflow_rejected = Some(true);
        row.overflow_rejected_without_mutation = Some(true);
    }

    if let Some(mut w) = logits {
        w.flush()
            .map_err(|_| Error::new("Failed to write logits trace"))?;
    }
    if let Some(mut w) = states {
        w.flush()
            .map_err(|_| Error::new("Failed to write state dump"))?;
    }
    check_persistent(session, persistent)?;

    row.position = session.position();
    row.positions = session.positions().to_vec();
    row.stable_storage = true;
    row.stable_kv_storage = true;
    row.prefill_calls = session.prefill_calls();
    row.decode_calls = session.decode_calls();
    Ok(row)
}

fn write_report(output: &Path, report: &ValidationReport) -> Result<()> {
    let mut file = BufWriter::new(write_output_file(&output.join("native.json"))?);
    let text = serde_json::to_string_pretty(report)?;
    writeln!(file, "{text}")
        .and_then(|()| file.flush())
        .map_err(|_| Error::new("Failed to write native validation report"))
}

/// Reads until `buf` is full or EOF so streaming file comparisons are immune to short reads.
fn read_full_chunk(file: &mut File, buf: &mut [u8]) -> Result<usize> {
    let mut total = 0;
    while total < buf.len() {
        let n = file
            .read(&mut buf[total..])
            .map_err(|_| Error::new("Parity trace read failed"))?;
        if n == 0 {
            break;
        }
        total += n;
    }
    Ok(total)
}

fn compare_files(actual: &Path, expected: &Path) -> Result<()> {
    let mut a = File::open(actual)
        .map_err(|_| Error::new(format!("Cannot open parity traces: {}", actual.display())))?;
    let mut b = File::open(expected)
        .map_err(|_| Error::new(format!("Cannot open parity traces: {}", actual.display())))?;
    let mut left = [0u8; 65536];
    let mut right = [0u8; 65536];
    loop {
        let na = read_full_chunk(&mut a, &mut left)?;
        let nb = read_full_chunk(&mut b, &mut right)?;
        require(
            na == nb && left[..na] == right[..nb],
            format!("Native bitwise mismatch: {}", actual.display()),
        )?;
        if na == 0 {
            break;
        }
    }
    Ok(())
}

fn check_parity(c: &Case, output: &Path, expected: &Path, row: &mut CaseReport) -> Result<()> {
    compare_files(
        &output.join(format!("{}.f32", c.name)),
        &expected.join(format!("{}.f32", c.name)),
    )?;
    for &step in &c.state_steps {
        let name = format!("{}.{}.states", c.name, step);
        compare_files(&output.join(&name), &expected.join(&name))?;
    }
    row.bitwise_parity = Some(true);
    // Keep failed traces; discard large native state dumps only after exact comparisons.
    for &step in &c.state_steps {
        let name = format!("{}.{}.states", c.name, step);
        let _ = fs::remove_file(output.join(name));
    }
    Ok(())
}

fn run_validation(options: &Options) -> Result<()> {
    let manifest = read_manifest(&options.cases)?;
    let trace_format = manifest.trace_format;
    require(
        trace_format == 1 || trace_format == 2,
        "Unsupported validation trace format",
    )?;
    let selected_states = trace_format == 2;
    let expected = if manifest.expected_directory.is_empty() {
        None
    } else {
        Some(PathBuf::from(&manifest.expected_directory))
    };

    let model = Model::open(&options.packages)?;
    let spec = model.spec().clone();
    let cases = parse_cases(manifest, &spec, &options.output)?;
    let layout = TraceLayout {
        separate_logits: spec.model_type == ModelType::Qwen3 || expected.is_some(),
        selected_states,
        compact: selected_states && spec.capacity > 1024,
    };
    require(
        expected.is_none() || (layout.separate_logits && selected_states),
        "Expected traces require separate logits and trace format 2",
    )?;

    let mut session = model.open_session(None)?;
    fs::create_dir_all(&options.output)?;

    let persistent: Vec<usize> = session
        .state()
        .iter()
        .map(|t| Ok(t.data_ptr()? as usize))
        .collect::<Result<_>>()?;

    let mut report = ValidationReport {
        cases: Vec::with_capacity(cases.len()),
        loader_pairs: 1,
        state_tensors: spec.state.len(),
        cache_buffers: (spec.model_type == ModelType::Qwen3).then_some(persistent.len()),
        no_python_library: None,
        attention_library: None,
        deltanet_library: None,
    };

    for c in &cases {
        let mut row = run_case(&mut session, c, layout, &options.output, &persistent)?;
        if let Some(exp_dir) = &expected {
            check_parity(c, &options.output, exp_dir, &mut row)?;
        }
        report.cases.push(row);
        if expected.is_some() {
            write_report(&options.output, &report)?;
        }
        eprintln!("{}: position {}", c.name, session.position());
    }

    check_native_libraries(&mut report, spec.model_type)?;
    write_report(&options.output, &report)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match parse_options(&args).and_then(|opts| run_validation(&opts)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
