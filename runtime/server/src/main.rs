//! Native OpenAI-compatible HTTP/SSE serving CLI (`xinfer_serve`).

use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use tokio::net::TcpListener;
use xinfer_engine::{Engine, EngineConfig};
use xinfer_runtime::{require, Error, Result};
use xinfer_server::build_router;

const USAGE: &str = "\
Usage: xinfer_serve <bundle_dir> [options]

Options:
  --host HOST               Bind address (default: 127.0.0.1)
  --port PORT               Bind port (default: 8000)
  --max-batch-size N        Allocated batch slots 1..=8 (default: bundle max)
  --max-queued-requests N   Max waiting requests before HTTP 429 (default: 64)
  --served-model-name NAME  Override model ID returned by /v1/models
  -h, --help                Show this help
";

struct CliOptions {
    bundle_dir: PathBuf,
    host: String,
    port: u16,
    engine: EngineConfig,
}

fn parse_usize(value: &str, flag: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| Error::invalid_input(format!("Invalid integer for {flag}: {value}")))
}

fn parse_options(args: &[String]) -> Result<CliOptions> {
    require(args.len() >= 2, "Missing bundle directory")?;
    let mut opts = CliOptions {
        bundle_dir: PathBuf::from(&args[1]),
        host: "127.0.0.1".to_string(),
        port: 8000,
        engine: EngineConfig::default(),
    };

    let mut i = 2;
    while i < args.len() {
        let flag = args[i].as_str();
        let next_val = |idx: usize| -> Result<&str> {
            args.get(idx + 1)
                .map(String::as_str)
                .ok_or_else(|| Error::invalid_input(format!("Missing option value: {flag}")))
        };
        match flag {
            "--host" => opts.host = next_val(i)?.to_string(),
            "--port" => {
                let raw = next_val(i)?;
                opts.port = raw
                    .parse::<u16>()
                    .map_err(|_| Error::invalid_input(format!("Invalid port: {raw}")))?;
            }
            "--max-batch-size" => {
                opts.engine.allocated_batch_size = Some(parse_usize(next_val(i)?, flag)?);
            }
            "--max-queued-requests" => {
                opts.engine.max_queued_requests = parse_usize(next_val(i)?, flag)?;
            }
            "--served-model-name" => {
                opts.engine.served_model_name = Some(next_val(i)?.to_string());
            }
            _ => return Err(Error::invalid_input(format!("Unknown option: {flag}"))),
        }
        i += 2;
    }
    Ok(opts)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

async fn run_server(opts: CliOptions) -> Result<()> {
    let addr: SocketAddr = format!("{}:{}", opts.host, opts.port)
        .parse()
        .map_err(|err| Error::invalid_input(format!("Invalid bind address: {err}")))?;

    eprintln!("Loading bundle from {} ...", opts.bundle_dir.display());
    let engine = Engine::start(&opts.bundle_dir, opts.engine)?;
    let handle = engine.handle();
    let info = handle.model_info().clone();
    eprintln!(
        "Ready: model={} type={} cache_length={} max_batch={} vocab={}",
        info.model_id,
        info.model_type.as_str(),
        info.cache_length,
        info.allocated_batch_size,
        info.vocab_size,
    );

    let router = build_router(Arc::new(handle));
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|err| Error::io(format!("Failed to bind {addr}: {err}")))?;
    eprintln!("Listening on http://{addr}");

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|err| Error::io(format!("HTTP server failed: {err}")))?;

    engine.shutdown();
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || matches!(args.get(1).map(String::as_str), Some("-h" | "--help")) {
        eprint!("{USAGE}");
        return if args.len() == 2 && matches!(args[1].as_str(), "-h" | "--help") {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }

    let opts = match parse_options(&args) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("error: Failed to start async runtime: {err}");
            return ExitCode::FAILURE;
        }
    };

    match rt.block_on(run_server(opts)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_options_accepts_minimal_flags_and_rejects_unknown() {
        let base = vec!["xinfer_serve".to_string(), "artifacts/bundle".to_string()];
        let opts = parse_options(&base).unwrap();
        assert_eq!(opts.host, "127.0.0.1");
        assert_eq!(opts.port, 8000);
        assert_eq!(opts.engine.allocated_batch_size, None);
        assert_eq!(opts.engine.prefill_chunk_size, None);
        assert_eq!(opts.engine.decode_steps_per_prefill_chunk, 1);

        let custom: Vec<String> = [
            "xinfer_serve",
            "artifacts/bundle",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--max-batch-size",
            "4",
            "--max-queued-requests",
            "16",
            "--served-model-name",
            "custom-qwen",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let opts = parse_options(&custom).unwrap();
        assert_eq!(opts.host, "0.0.0.0");
        assert_eq!(opts.port, 9000);
        assert_eq!(opts.engine.allocated_batch_size, Some(4));
        assert_eq!(opts.engine.max_queued_requests, 16);
        assert_eq!(
            opts.engine.served_model_name.as_deref(),
            Some("custom-qwen")
        );

        let unknown: Vec<String> = [
            "xinfer_serve",
            "artifacts/bundle",
            "--prefill-chunk-size",
            "256",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert!(parse_options(&unknown)
            .err()
            .unwrap()
            .to_string()
            .contains("Unknown option"));
    }
}
