//! Dedicated GPU actor thread owning `xinfer_runtime::Session` (`!Send + !Sync`)
//! and a cloneable async `EngineHandle` for callers such as `xinfer_serve`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use tokio::sync::{mpsc, oneshot};
use xinfer_runtime::{require_kind, Error, ErrorKind, Model, Result};

use crate::config::EngineConfig;
use crate::scheduler::{PendingRequest, Scheduler, SessionBackend};
use crate::tokenizer::BundleTokenizer;
use crate::types::{ChatMessage, GenerateEvent, GenerateRequest, ModelInfo, SamplingParams};

const EVENT_CHANNEL_CAPACITY: usize = 64;

enum Command {
    Submit {
        pending: PendingRequest,
        ack: oneshot::Sender<Result<()>>,
    },
    Shutdown,
}

struct StartupInfo {
    model_info: ModelInfo,
    tokenizer: Arc<BundleTokenizer>,
}

/// Cloneable, thread-safe handle to a running inference engine.
#[derive(Clone)]
pub struct EngineHandle {
    cmd_tx: mpsc::Sender<Command>,
    model_info: Arc<ModelInfo>,
    tokenizer: Arc<BundleTokenizer>,
    ready: Arc<AtomicBool>,
}

impl EngineHandle {
    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    pub fn tokenizer(&self) -> &Arc<BundleTokenizer> {
        &self.tokenizer
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire) && !self.cmd_tx.is_closed()
    }

    /// Tokenizes `prompt` on the caller's thread and submits a completion request to the GPU actor.
    pub async fn submit_completion(
        &self,
        prompt: &str,
        max_tokens: Option<usize>,
        sampling: SamplingParams,
        stop_strings: Vec<String>,
    ) -> Result<mpsc::Receiver<GenerateEvent>> {
        let prompt_tokens = self.tokenizer.encode(prompt, true)?;
        let resolved_max = self.resolve_max_tokens(prompt_tokens.len(), max_tokens)?;
        self.submit_tokens(GenerateRequest {
            prompt_tokens,
            max_tokens: resolved_max,
            sampling,
            stop_strings,
            stop_token_ids: Vec::new(),
            ignore_eos: false,
        })
        .await
    }

    /// Renders the bundled chat template and tokenizes on the caller's thread, then submits to the GPU actor.
    pub async fn submit_chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        sampling: SamplingParams,
        stop_strings: Vec<String>,
        enable_thinking: Option<bool>,
    ) -> Result<mpsc::Receiver<GenerateEvent>> {
        let rendered = self
            .tokenizer
            .apply_chat_template(messages, true, enable_thinking)?;
        // Template already inserts special tokens such as `<|im_start|>`.
        let prompt_tokens = self.tokenizer.encode(&rendered, false)?;
        let resolved_max = self.resolve_max_tokens(prompt_tokens.len(), max_tokens)?;
        self.submit_tokens(GenerateRequest {
            prompt_tokens,
            max_tokens: resolved_max,
            sampling,
            stop_strings,
            stop_token_ids: Vec::new(),
            ignore_eos: false,
        })
        .await
    }

    /// Submits pre-tokenized IDs directly (useful for exact greedy parity tests vs `xinfer_aoti`).
    pub async fn submit_tokens(
        &self,
        request: GenerateRequest,
    ) -> Result<mpsc::Receiver<GenerateEvent>> {
        if !self.is_ready() {
            return Err(Error::new("Engine is not ready"));
        }
        request.sampling.validate()?;
        require_kind(
            !request.prompt_tokens.is_empty(),
            ErrorKind::InvalidInput,
            "Prompt must contain at least 1 token",
        )?;
        require_kind(
            request.max_tokens >= 1,
            ErrorKind::InvalidInput,
            "max_tokens must be >= 1",
        )?;
        let vocab = self.model_info.vocab_size as i64;
        for &id in &request.prompt_tokens {
            require_kind(
                id >= 0 && id < vocab,
                ErrorKind::InvalidInput,
                "Token ID out of range",
            )?;
        }
        let total = request
            .prompt_tokens
            .len()
            .saturating_add(request.max_tokens);
        require_kind(
            total <= self.model_info.cache_length,
            ErrorKind::CapacityExceeded,
            format!(
                "Prompt ({}) + max_tokens ({}) exceeds context capacity ({})",
                request.prompt_tokens.len(),
                request.max_tokens,
                self.model_info.cache_length
            ),
        )?;

        let (event_tx, event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let (ack_tx, ack_rx) = oneshot::channel();
        self.cmd_tx
            .try_send(Command::Submit {
                pending: PendingRequest {
                    request,
                    tx: event_tx,
                },
                ack: ack_tx,
            })
            .map_err(|err| match err {
                mpsc::error::TrySendError::Full(_) => {
                    Error::capacity_exceeded("Request queue is full")
                }
                mpsc::error::TrySendError::Closed(_) => Error::new("Engine is shutting down"),
            })?;

        let result = ack_rx
            .await
            .map_err(|_| Error::new("Engine terminated before acknowledging request"))?;
        result?;
        Ok(event_rx)
    }

    fn resolve_max_tokens(&self, prompt_len: usize, requested: Option<usize>) -> Result<usize> {
        require_kind(
            prompt_len > 0,
            ErrorKind::InvalidInput,
            "Prompt must contain at least 1 token",
        )?;
        let remaining = self.model_info.cache_length.saturating_sub(prompt_len);
        require_kind(
            remaining >= 1,
            ErrorKind::CapacityExceeded,
            format!(
                "Prompt ({prompt_len}) leaves no room in context capacity ({})",
                self.model_info.cache_length
            ),
        )?;
        match requested {
            Some(limit) => Ok(limit),
            None => Ok(remaining),
        }
    }

    pub fn request_shutdown(&self) {
        self.ready.store(false, Ordering::Release);
        let _ = self.cmd_tx.try_send(Command::Shutdown);
    }
}

/// Owns the dedicated GPU actor thread and joins it cleanly on shutdown/drop.
pub struct Engine {
    handle: EngineHandle,
    worker: Option<JoinHandle<()>>,
}

impl Engine {
    /// Opens the bundle on a dedicated `"xinfer-gpu"` OS thread, validates metadata/checksums,
    /// loads weights/packages/tokenizer, and returns once ready to serve.
    pub fn start(bundle_dir: impl AsRef<Path>, config: EngineConfig) -> Result<Self> {
        let bundle_dir: PathBuf = bundle_dir.as_ref().to_path_buf();
        let queue_bound = config.max_queued_requests.max(1) + 16;
        let (cmd_tx, cmd_rx) = mpsc::channel(queue_bound);
        let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<StartupInfo>>(1);
        let ready = Arc::new(AtomicBool::new(false));
        let ready_worker = Arc::clone(&ready);

        let worker = thread::Builder::new()
            .name("xinfer-gpu".to_string())
            .spawn(move || {
                run_gpu_actor(bundle_dir, config, cmd_rx, ready_tx, ready_worker);
            })
            .map_err(|err| Error::io(format!("Failed to spawn GPU worker thread: {err}")))?;

        let startup = ready_rx
            .recv()
            .map_err(|_| Error::new("GPU worker exited unexpectedly during startup"))??;

        let handle = EngineHandle {
            cmd_tx,
            model_info: Arc::new(startup.model_info),
            tokenizer: startup.tokenizer,
            ready,
        };
        Ok(Self {
            handle,
            worker: Some(worker),
        })
    }

    pub fn handle(&self) -> EngineHandle {
        self.handle.clone()
    }

    pub fn shutdown(mut self) {
        self.handle.request_shutdown();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.handle.request_shutdown();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn init_actor(
    bundle_dir: &Path,
    config: &EngineConfig,
) -> Result<(StartupInfo, Scheduler<SessionBackend>)> {
    let model = Model::open(bundle_dir)?;
    let spec = model.spec().clone();
    let resolved = config.resolve(&spec)?;
    let session = model.open_session(Some(resolved.allocated_batch_size as i64))?;
    session.synchronize()?;

    let tokenizer = Arc::new(BundleTokenizer::load(session.bundle_dir(), &spec.eos)?);
    let default_eos = tokenizer.eos_token_ids().to_vec();
    let model_info = ModelInfo {
        model_id: resolved.served_model_name.clone(),
        model_type: spec.model_type,
        cache_length: spec.capacity as usize,
        max_prefill_tokens: spec.max_prefill as usize,
        allocated_batch_size: resolved.allocated_batch_size,
        vocab_size: spec.vocab as usize,
    };
    let scheduler = Scheduler::new(
        SessionBackend::new(session),
        Some(Arc::clone(&tokenizer)),
        default_eos,
        resolved,
    );
    Ok((
        StartupInfo {
            model_info,
            tokenizer,
        },
        scheduler,
    ))
}

fn run_gpu_actor(
    bundle_dir: PathBuf,
    config: EngineConfig,
    mut cmd_rx: mpsc::Receiver<Command>,
    ready_tx: std_mpsc::SyncSender<Result<StartupInfo>>,
    ready: Arc<AtomicBool>,
) {
    let mut scheduler = match init_actor(&bundle_dir, &config) {
        Ok((startup, scheduler)) => {
            ready.store(true, Ordering::Release);
            let _ = ready_tx.send(Ok(startup));
            scheduler
        }
        Err(err) => {
            let _ = ready_tx.send(Err(err));
            return;
        }
    };

    let mut shutting_down = false;
    while !shutting_down {
        if !scheduler.has_work() {
            match cmd_rx.blocking_recv() {
                Some(Command::Submit { pending, ack }) => {
                    let _ = ack.send(scheduler.submit(pending));
                }
                Some(Command::Shutdown) | None => break,
            }
        }

        // Drain any newly arrived requests before launching the next GPU step.
        loop {
            match cmd_rx.try_recv() {
                Ok(Command::Submit { pending, ack }) => {
                    let _ = ack.send(scheduler.submit(pending));
                }
                Ok(Command::Shutdown) => {
                    shutting_down = true;
                    break;
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    shutting_down = true;
                    break;
                }
            }
        }
        if shutting_down {
            break;
        }

        if let Err(err) = scheduler.step() {
            ready.store(false, Ordering::Release);
            scheduler.abort_all(&format!("Inference engine error: {err}"));
            return;
        }
    }

    ready.store(false, Ordering::Release);
    scheduler.abort_all("Engine shutting down");
}
