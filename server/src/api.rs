//! Minimal OpenAI-compatible HTTP/JSON and SSE request/response schemas and error mapping.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use xinfer_engine::{ChatMessage, SamplingParams, Usage};
use xinfer_runtime::{Error, ErrorKind};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopParam {
    Single(String),
    Multiple(Vec<String>),
}

impl StopParam {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multiple(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub seed: Option<u64>,
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct ChatTemplateKwargs {
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub seed: Option<u64>,
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub stream: bool,
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
}

pub fn sampling_from_openai(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<u64>,
) -> SamplingParams {
    SamplingParams {
        temperature: temperature.unwrap_or(1.0),
        top_p: top_p.unwrap_or(1.0),
        top_k: top_k.unwrap_or(0),
        seed,
    }
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    pub max_model_len: usize,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub struct UsageResponse {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl From<Usage> for UsageResponse {
    fn from(usage: Usage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageResponse>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: UsageResponse,
}

#[derive(Debug, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatStreamChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    code: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    error_type: &'static str,
    code: Option<&'static str>,
    message: String,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            code: None,
            message: message.into(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error_type: "invalid_request_error",
            code: Some("model_not_found"),
            message: message.into(),
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            error_type: "service_unavailable",
            code: None,
            message: message.into(),
        }
    }

    pub fn from_runtime_error(err: Error) -> Self {
        let message = err.to_string();
        match err.kind() {
            ErrorKind::InvalidInput | ErrorKind::Unsupported => Self::bad_request(message),
            ErrorKind::CapacityExceeded => {
                if message.contains("queue") {
                    Self {
                        status: StatusCode::TOO_MANY_REQUESTS,
                        error_type: "rate_limit_error",
                        code: Some("queue_full"),
                        message,
                    }
                } else {
                    Self {
                        status: StatusCode::BAD_REQUEST,
                        error_type: "invalid_request_error",
                        code: Some("context_length_exceeded"),
                        message,
                    }
                }
            }
            ErrorKind::Io | ErrorKind::Runtime => {
                if message.contains("not ready") || message.contains("shutting down") {
                    Self::service_unavailable(message)
                } else {
                    Self {
                        status: StatusCode::INTERNAL_SERVER_ERROR,
                        error_type: "server_error",
                        code: None,
                        message,
                    }
                }
            }
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorEnvelope {
            error: ErrorBody {
                message: self.message,
                error_type: self.error_type,
                code: self.code,
            },
        };
        (self.status, Json(body)).into_response()
    }
}
