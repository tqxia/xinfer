//! Minimal OpenAI-compatible HTTP/SSE router layered over `xinfer-engine`.

pub mod api;

use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use tokio::sync::mpsc;
use xinfer_engine::{
    ChatMessage, EngineHandle, FinishReason, GenerateEvent, ModelInfo, SamplingParams, Usage,
};
use xinfer_runtime::Result;

use crate::api::{
    sampling_from_openai, ApiError, ChatCompletionChoice, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, ChatDelta, ChatStreamChoice, CompletionChoice,
    CompletionRequest, CompletionResponse, ModelCard, ModelListResponse, UsageResponse,
};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Minimal async backend trait implemented by `EngineHandle` and mock test doubles.
pub trait ServingBackend: Send + Sync + 'static {
    fn model_info(&self) -> &ModelInfo;
    fn is_ready(&self) -> bool;
    fn submit_completion<'a>(
        &'a self,
        prompt: &'a str,
        max_tokens: Option<usize>,
        sampling: SamplingParams,
        stop_strings: Vec<String>,
    ) -> BoxFuture<'a, Result<mpsc::Receiver<GenerateEvent>>>;
    fn submit_chat<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        max_tokens: Option<usize>,
        sampling: SamplingParams,
        stop_strings: Vec<String>,
        enable_thinking: Option<bool>,
    ) -> BoxFuture<'a, Result<mpsc::Receiver<GenerateEvent>>>;
}

impl ServingBackend for EngineHandle {
    fn model_info(&self) -> &ModelInfo {
        EngineHandle::model_info(self)
    }

    fn is_ready(&self) -> bool {
        EngineHandle::is_ready(self)
    }

    fn submit_completion<'a>(
        &'a self,
        prompt: &'a str,
        max_tokens: Option<usize>,
        sampling: SamplingParams,
        stop_strings: Vec<String>,
    ) -> BoxFuture<'a, Result<mpsc::Receiver<GenerateEvent>>> {
        Box::pin(EngineHandle::submit_completion(
            self,
            prompt,
            max_tokens,
            sampling,
            stop_strings,
        ))
    }

    fn submit_chat<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        max_tokens: Option<usize>,
        sampling: SamplingParams,
        stop_strings: Vec<String>,
        enable_thinking: Option<bool>,
    ) -> BoxFuture<'a, Result<mpsc::Receiver<GenerateEvent>>> {
        Box::pin(EngineHandle::submit_chat(
            self,
            messages,
            max_tokens,
            sampling,
            stop_strings,
            enable_thinking,
        ))
    }
}

struct AppState {
    backend: Arc<dyn ServingBackend>,
    started_at: u64,
    next_id: AtomicU64,
}

impl AppState {
    fn new_request_id(&self, prefix: &str) -> String {
        let seq = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{}-{seq}", unix_timestamp())
    }
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Builds the minimal v1 HTTP router (`/health`, `/ready`, `/v1/models`,
/// `/v1/completions`, `/v1/chat/completions`).
pub fn build_router(backend: Arc<dyn ServingBackend>) -> Router {
    let state = Arc::new(AppState {
        backend,
        started_at: unix_timestamp(),
        next_id: AtomicU64::new(1),
    });
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn ready(State(state): State<Arc<AppState>>) -> Response {
    if state.backend.is_ready() {
        StatusCode::OK.into_response()
    } else {
        ApiError::service_unavailable("Engine is not ready").into_response()
    }
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let info = state.backend.model_info();
    Json(ModelListResponse {
        object: "list",
        data: vec![ModelCard {
            id: info.model_id.clone(),
            object: "model",
            created: state.started_at,
            owned_by: "xinfer",
            max_model_len: info.cache_length,
        }],
    })
}

fn check_model_name(requested: Option<&str>, served: &str) -> std::result::Result<(), ApiError> {
    if let Some(name) = requested {
        if !name.is_empty() && name != served {
            return Err(ApiError::not_found(format!(
                "Model '{name}' not found; serving '{served}'"
            )));
        }
    }
    Ok(())
}

async fn collect_all(
    mut rx: mpsc::Receiver<GenerateEvent>,
) -> std::result::Result<(String, FinishReason, Usage), ApiError> {
    let mut output = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            GenerateEvent::Delta { text, .. } => output.push_str(&text),
            GenerateEvent::Finished { reason, usage } => return Ok((output, reason, usage)),
            GenerateEvent::Error(err) => return Err(ApiError::from_runtime_error(err)),
        }
    }
    Err(ApiError::service_unavailable(
        "Generation stream closed unexpectedly",
    ))
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> std::result::Result<Response, ApiError> {
    let model = state.backend.model_info().model_id.clone();
    check_model_name(req.model.as_deref(), &model)?;
    if req.prompt.is_empty() {
        return Err(ApiError::bad_request("prompt must not be empty"));
    }

    let sampling = sampling_from_openai(req.temperature, req.top_p, req.top_k, req.seed);
    let stop = req.stop.map(|s| s.into_vec()).unwrap_or_default();
    let rx = state
        .backend
        .submit_completion(&req.prompt, req.max_tokens, sampling, stop)
        .await
        .map_err(ApiError::from_runtime_error)?;

    let id = state.new_request_id("cmpl");
    let created = unix_timestamp();

    if !req.stream {
        let (text, reason, usage) = collect_all(rx).await?;
        return Ok(Json(CompletionResponse {
            id,
            object: "text_completion",
            created,
            model,
            choices: vec![CompletionChoice {
                index: 0,
                text,
                finish_reason: Some(reason.as_openai_str()),
            }],
            usage: Some(UsageResponse::from(usage)),
        })
        .into_response());
    }

    let sse_stream = stream::unfold((rx, false), move |(mut rx, done)| {
        let id = id.clone();
        let model = model.clone();
        async move {
            if done {
                return None;
            }
            match rx.recv().await {
                Some(GenerateEvent::Delta { text, .. }) => {
                    if text.is_empty() {
                        // Emit an empty keep-alive-free skip? Keep folding by returning an event
                        // only when text is non-empty is cleaner, so loop inside unfold.
                        let mut rx = rx;
                        loop {
                            match rx.recv().await {
                                Some(GenerateEvent::Delta { text, .. }) if text.is_empty() => {
                                    continue
                                }
                                Some(GenerateEvent::Delta { text, .. }) => {
                                    let payload = serde_json::to_string(&CompletionResponse {
                                        id,
                                        object: "text_completion",
                                        created,
                                        model,
                                        choices: vec![CompletionChoice {
                                            index: 0,
                                            text,
                                            finish_reason: None,
                                        }],
                                        usage: None,
                                    })
                                    .unwrap_or_default();
                                    return Some((
                                        Ok::<Event, Infallible>(Event::default().data(payload)),
                                        (rx, false),
                                    ));
                                }
                                Some(GenerateEvent::Finished { reason, .. }) => {
                                    let payload = serde_json::to_string(&CompletionResponse {
                                        id,
                                        object: "text_completion",
                                        created,
                                        model,
                                        choices: vec![CompletionChoice {
                                            index: 0,
                                            text: String::new(),
                                            finish_reason: Some(reason.as_openai_str()),
                                        }],
                                        usage: None,
                                    })
                                    .unwrap_or_default();
                                    return Some((
                                        Ok::<Event, Infallible>(Event::default().data(payload)),
                                        (rx, true),
                                    ));
                                }
                                Some(GenerateEvent::Error(_)) | None => {
                                    return Some((
                                        Ok::<Event, Infallible>(Event::default().data("[DONE]")),
                                        (rx, true),
                                    ));
                                }
                            }
                        }
                    }
                    let payload = serde_json::to_string(&CompletionResponse {
                        id,
                        object: "text_completion",
                        created,
                        model,
                        choices: vec![CompletionChoice {
                            index: 0,
                            text,
                            finish_reason: None,
                        }],
                        usage: None,
                    })
                    .unwrap_or_default();
                    Some((
                        Ok::<Event, Infallible>(Event::default().data(payload)),
                        (rx, false),
                    ))
                }
                Some(GenerateEvent::Finished { reason, .. }) => {
                    let payload = serde_json::to_string(&CompletionResponse {
                        id,
                        object: "text_completion",
                        created,
                        model,
                        choices: vec![CompletionChoice {
                            index: 0,
                            text: String::new(),
                            finish_reason: Some(reason.as_openai_str()),
                        }],
                        usage: None,
                    })
                    .unwrap_or_default();
                    // Send final chunk, then [DONE] on the next poll by transitioning state.
                    Some((
                        Ok::<Event, Infallible>(Event::default().data(payload)),
                        (rx, true),
                    ))
                }
                Some(GenerateEvent::Error(_)) | None => None,
            }
        }
    });

    let done_stream =
        stream::once(async { Ok::<Event, Infallible>(Event::default().data("[DONE]")) });
    let combined = futures_util::StreamExt::chain(sse_stream, done_stream);
    Ok(Sse::new(combined)
        .keep_alive(KeepAlive::default())
        .into_response())
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> std::result::Result<Response, ApiError> {
    let model = state.backend.model_info().model_id.clone();
    check_model_name(req.model.as_deref(), &model)?;
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }

    let max_tokens = req.max_completion_tokens.or(req.max_tokens);
    let sampling = sampling_from_openai(req.temperature, req.top_p, req.top_k, req.seed);
    let stop = req.stop.map(|s| s.into_vec()).unwrap_or_default();
    let enable_thinking = req.chat_template_kwargs.and_then(|kw| kw.enable_thinking);

    let rx = state
        .backend
        .submit_chat(&req.messages, max_tokens, sampling, stop, enable_thinking)
        .await
        .map_err(ApiError::from_runtime_error)?;

    let id = state.new_request_id("chatcmpl");
    let created = unix_timestamp();

    if !req.stream {
        let (content, reason, usage) = collect_all(rx).await?;
        return Ok(Json(ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: reason.as_openai_str(),
            }],
            usage: UsageResponse::from(usage),
        })
        .into_response());
    }

    let role_chunk = serde_json::to_string(&ChatCompletionChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model.clone(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant"),
                content: Some(String::new()),
            },
            finish_reason: None,
        }],
    })
    .unwrap_or_default();

    let first_stream =
        stream::once(async move { Ok::<Event, Infallible>(Event::default().data(role_chunk)) });

    let body_stream = stream::unfold((rx, false), move |(mut rx, done)| {
        let id = id.clone();
        let model = model.clone();
        async move {
            if done {
                return None;
            }
            loop {
                match rx.recv().await {
                    Some(GenerateEvent::Delta { text, .. }) if text.is_empty() => continue,
                    Some(GenerateEvent::Delta { text, .. }) => {
                        let payload = serde_json::to_string(&ChatCompletionChunk {
                            id,
                            object: "chat.completion.chunk",
                            created,
                            model,
                            choices: vec![ChatStreamChoice {
                                index: 0,
                                delta: ChatDelta {
                                    role: None,
                                    content: Some(text),
                                },
                                finish_reason: None,
                            }],
                        })
                        .unwrap_or_default();
                        return Some((
                            Ok::<Event, Infallible>(Event::default().data(payload)),
                            (rx, false),
                        ));
                    }
                    Some(GenerateEvent::Finished { reason, .. }) => {
                        let payload = serde_json::to_string(&ChatCompletionChunk {
                            id,
                            object: "chat.completion.chunk",
                            created,
                            model,
                            choices: vec![ChatStreamChoice {
                                index: 0,
                                delta: ChatDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some(reason.as_openai_str()),
                            }],
                        })
                        .unwrap_or_default();
                        return Some((
                            Ok::<Event, Infallible>(Event::default().data(payload)),
                            (rx, true),
                        ));
                    }
                    Some(GenerateEvent::Error(_)) | None => return None,
                }
            }
        }
    });

    let done_stream =
        stream::once(async { Ok::<Event, Infallible>(Event::default().data("[DONE]")) });
    let combined = futures_util::StreamExt::chain(
        futures_util::StreamExt::chain(first_stream, body_stream),
        done_stream,
    );
    Ok(Sse::new(combined)
        .keep_alive(KeepAlive::default())
        .into_response())
}

#[cfg(test)]
mod tests;
