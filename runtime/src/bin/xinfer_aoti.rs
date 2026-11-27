//! Standalone greedy-generation CLI (`xinfer_aoti`).
//!
//! Validates package metadata, prompt token IDs, output/trace paths (refusing
//! overwrites and dangling symlinks), runs greedy generation with `allocated_batch_size = 1`,
//! and prints timing/stop metrics to stderr.

use std::env;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use xinfer_runtime::{
    create_output_file, cuda_synchronize, parse_integer, path_exists, require, weakly_canonical,
    Error, Model, Result, Spec,
};

const USAGE: &str =
    "Usage: xinfer_aoti PACKAGE_DIR PROMPT_IDS OUTPUT_IDS [--max-output-tokens N] [--trace LOGITS_F32]\n";

struct Options {
    packages: PathBuf,
    input: PathBuf,
    output: PathBuf,
    trace: Option<PathBuf>,
    limit: i64,
}

fn parse_options(args: &[String]) -> Result<Options> {
    let mut options = Options {
        packages: PathBuf::from(&args[1]),
        input: PathBuf::from(&args[2]),
        output: PathBuf::from(&args[3]),
        trace: None,
        limit: 64,
    };

    let mut i = 4;
    while i < args.len() {
        require(i + 1 < args.len(), "Missing option value")?;
        let option = args[i].as_str();
        let value = args[i + 1].as_str();
        match option {
            "--max-output-tokens" => options.limit = parse_integer(value)?,
            "--trace" => options.trace = Some(PathBuf::from(value)),
            _ => return Err(Error::new(format!("Unknown option: {option}"))),
        }
        i += 2;
    }

    Ok(options)
}

fn validate_request(options: &Options, spec: &Spec) -> Result<Vec<i64>> {
    let prompt = spec.read_ids(&options.input)?;
    spec.check_request(&prompt, options.limit)?;

    require(!path_exists(&options.output), "Output already exists")?;
    let input_path = weakly_canonical(&options.input)?;
    let canonical_output = weakly_canonical(&options.output)?;
    require(
        input_path != canonical_output,
        "Input and output paths must differ",
    )?;
    if let Some(trace) = &options.trace {
        let canonical_trace = weakly_canonical(trace)?;
        require(
            canonical_trace != input_path && canonical_trace != canonical_output,
            "Trace path must differ from input and output",
        )?;
        require(!path_exists(trace), "Trace path already exists")?;
    }
    Ok(prompt)
}

fn write_generated_ids(path: &Path, generated: &[i64]) -> Result<()> {
    let mut writer = BufWriter::new(create_output_file(path)?);
    for &id in generated {
        write!(writer, "{id} ").map_err(|_| Error::new("Failed to write generated IDs"))?;
    }
    writeln!(writer).map_err(|_| Error::new("Failed to write generated IDs"))?;
    writer
        .flush()
        .map_err(|_| Error::new("Failed to write generated IDs"))
}

fn run_generation(options: &Options) -> Result<()> {
    let model = Model::open(&options.packages)?;
    let prompt = validate_request(options, model.spec())?;

    let load_start = Instant::now();
    let mut session = model.open_session(Some(1))?;
    cuda_synchronize()?;

    let run_start = Instant::now();
    let mut trace_writer = match &options.trace {
        Some(path) => Some(BufWriter::new(create_output_file(path)?)),
        None => None,
    };

    let generated = session.generate(
        &prompt,
        options.limit,
        trace_writer.as_mut().map(|w| w as &mut dyn Write),
    )?;
    cuda_synchronize()?;
    let run_end = Instant::now();

    write_generated_ids(&options.output, &generated)?;
    if let Some(mut writer) = trace_writer {
        writer
            .flush()
            .map_err(|_| Error::new("Failed to write logits trace"))?;
    }

    let load_ms = run_start.duration_since(load_start).as_secs_f64() * 1000.0;
    let generation_ms = run_end.duration_since(run_start).as_secs_f64() * 1000.0;
    let stop = if generated
        .last()
        .map(|last| model.spec().eos.contains(last))
        .unwrap_or(false)
    {
        "eos"
    } else {
        "limit"
    };
    let trace_flag = if options.trace.is_some() { 1 } else { 0 };
    eprintln!(
        "prompt_tokens={} generated_tokens={} consumed_positions={} stop={} load_ms={} generation_ms={} trace={}",
        prompt.len(),
        generated.len(),
        session.position(),
        stop,
        load_ms,
        generation_ms,
        trace_flag,
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.get(1).map(String::as_str) == Some("--help") {
        eprint!("{USAGE}");
        return if args.len() == 2 && args[1] == "--help" {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }

    match parse_options(&args).and_then(|opts| run_generation(&opts)) {
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
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!("xinfer-cli-test-{tag}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn dangling_symlink_is_detected() {
        let dir = temp_dir("dangling");
        let missing = dir.join("missing_target");
        let link = dir.join("dangling_output");
        symlink(&missing, &link).unwrap();

        assert!(!path_exists(&missing));
        assert!(!link.exists());
        assert!(path_exists(&link));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_options_rejects_unknown_and_missing_value() {
        let base = vec![
            "xinfer_aoti".to_string(),
            "pkg".to_string(),
            "in.ids".to_string(),
            "out.ids".to_string(),
        ];
        assert!(parse_options(&base).is_ok());

        let mut missing = base.clone();
        missing.push("--trace".to_string());
        assert!(parse_options(&missing)
            .err()
            .unwrap()
            .to_string()
            .contains("Missing option value"));

        let mut unknown = base.clone();
        unknown.push("--max-new-tokens".to_string());
        unknown.push("4".to_string());
        assert!(parse_options(&unknown)
            .err()
            .unwrap()
            .to_string()
            .contains("Unknown option"));
    }

    #[test]
    fn weakly_canonical_resolves_future_paths() {
        let dir = temp_dir("canonical");
        let nested = dir.join("sub").join("..").join("out.ids");
        let direct = dir.join("out.ids");
        assert_eq!(
            weakly_canonical(&nested).unwrap(),
            weakly_canonical(&direct).unwrap()
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
