use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use xinfer_engine::{ChatMessage, FinishReason, GenerateEvent, ModelInfo, SamplingParams, Usage};
use xinfer_runtime::{Error, ModelType, Result};

use crate::{build_router, BoxFuture, ServingBackend};

struct MockBackend {
    info: ModelInfo,
    ready: AtomicBool,
    fail_with: Option<fn() -> Error>,
}

impl MockBackend {
    fn new() -> Self {
        Self {
            info: ModelInfo {
                model_id: "Qwen/Mock-0.8B".to_string(),
                model_type: ModelType::Qwen35,
                cache_length: 1024,
                max_prefill_tokens: 512,
                allocated_batch_size: 4,
                vocab_size: 1000,
            },
            ready: AtomicBool::new(true),
            fail_with: None,
        }
    }
}

impl ServingBackend for MockBackend {
    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    fn submit_completion<'a>(
        &'a self,
        prompt: &'a str,
        _max_tokens: Option<usize>,
        _sampling: SamplingParams,
        _stop_strings: Vec<String>,
    ) -> BoxFuture<'a, Result<mpsc::Receiver<GenerateEvent>>> {
        let fail = self.fail_with;
        let prompt_len = prompt.len();
        Box::pin(async move {
            if let Some(make_err) = fail {
                return Err(make_err());
            }
            let (tx, rx) = mpsc::channel(8);
            tokio::spawn(async move {
                let _ = tx
                    .send(GenerateEvent::Delta {
                        token_id: 1,
                        text: "hello ".to_string(),
                    })
                    .await;
                let _ = tx
                    .send(GenerateEvent::Delta {
                        token_id: 2,
                        text: "world".to_string(),
                    })
                    .await;
                let _ = tx
                    .send(GenerateEvent::Finished {
                        reason: FinishReason::Stop,
                        usage: Usage {
                            prompt_tokens: prompt_len,
                            completion_tokens: 2,
                        },
                    })
                    .await;
            });
            Ok(rx)
        })
    }

    fn submit_chat<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        _max_tokens: Option<usize>,
        _sampling: SamplingParams,
        _stop_strings: Vec<String>,
        enable_thinking: Option<bool>,
    ) -> BoxFuture<'a, Result<mpsc::Receiver<GenerateEvent>>> {
        let fail = self.fail_with;
        let count = messages.len();
        Box::pin(async move {
            if let Some(make_err) = fail {
                return Err(make_err());
            }
            let (tx, rx) = mpsc::channel(8);
            tokio::spawn(async move {
                let tag = if enable_thinking == Some(false) {
                    "no-think"
                } else {
                    "ok"
                };
                let _ = tx
                    .send(GenerateEvent::Delta {
                        token_id: 10,
                        text: format!("{tag}:{count}"),
                    })
                    .await;
                let _ = tx
                    .send(GenerateEvent::Finished {
                        reason: FinishReason::Stop,
                        usage: Usage {
                            prompt_tokens: 5,
                            completion_tokens: 1,
                        },
                    })
                    .await;
            });
            Ok(rx)
        })
    }
}

async fn send_http(addr: std::net::SocketAddr, req: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.unwrap();
    let text = String::from_utf8(raw).unwrap();
    let status_line = text.lines().next().unwrap();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let body = text.split_once("\r\n\r\n").unwrap().1.to_string();
    (status, body)
}

#[tokio::test]
async fn endpoints_streaming_and_error_mapping() {
    let backend = Arc::new(MockBackend::new());
    let router = build_router(backend.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // /health & /ready
    let (code, _) = send_http(
        addr,
        "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert_eq!(code, 200);
    let (code, _) = send_http(
        addr,
        "GET /ready HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert_eq!(code, 200);

    // /v1/models
    let (code, body) = send_http(
        addr,
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert_eq!(code, 200);
    let models: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(models["data"][0]["id"], "Qwen/Mock-0.8B");
    assert_eq!(models["data"][0]["max_model_len"], 1024);

    // Non-streaming /v1/completions
    let payload = r#"{"model":"Qwen/Mock-0.8B","prompt":"hi","max_tokens":4}"#;
    let req = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    let (code, body) = send_http(addr, &req).await;
    assert_eq!(code, 200);
    let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(resp["choices"][0]["text"], "hello world");
    assert_eq!(resp["choices"][0]["finish_reason"], "stop");
    assert_eq!(resp["usage"]["completion_tokens"], 2);

    // Streaming /v1/chat/completions (SSE)
    let payload = r#"{"model":"Qwen/Mock-0.8B","messages":[{"role":"user","content":"hi"}],"stream":true,"chat_template_kwargs":{"enable_thinking":false}}"#;
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    let (code, body) = send_http(addr, &req).await;
    assert_eq!(code, 200);
    assert!(body.contains(r#""role":"assistant""#));
    assert!(body.contains(r#""content":"no-think:1""#));
    assert!(body.contains(r#""finish_reason":"stop""#));
    assert!(body.contains("data: [DONE]"));

    // Unknown model -> 404
    let payload = r#"{"model":"wrong-model","prompt":"hi"}"#;
    let req = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    let (code, body) = send_http(addr, &req).await;
    assert_eq!(code, 404);
    assert!(body.contains("model_not_found"));

    // Not ready -> 503
    backend.ready.store(false, Ordering::Release);
    let (code, _) = send_http(
        addr,
        "GET /ready HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert_eq!(code, 503);

    server.abort();
}

#[tokio::test]
async fn queue_full_maps_to_429() {
    let mut mock = MockBackend::new();
    mock.fail_with = Some(|| Error::capacity_exceeded("Request queue is full"));
    let router = build_router(Arc::new(mock));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let payload = r#"{"prompt":"hi"}"#;
    let req = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    let (code, body) = send_http(addr, &req).await;
    assert_eq!(code, 429);
    assert!(body.contains("queue_full"));

    server.abort();
}
