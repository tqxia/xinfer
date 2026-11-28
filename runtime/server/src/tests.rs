use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use xinfer_engine::{ChatMessage, FinishReason, GenerateEvent, GenerateOptions, ModelInfo, Usage};
use xinfer_runtime::{Error, ModelType, Result};

use crate::{build_router, BoxFuture, ServingBackend};

enum MockStreamMode {
    Normal,
    EmptyDeltaThenFinished,
    EmptyDeltaThenError,
    MidStreamUnavailableError,
}

struct MockBackend {
    info: ModelInfo,
    ready: AtomicBool,
    fail_with: Option<fn() -> Error>,
    stream_mode: MockStreamMode,
    last_options: Mutex<Option<GenerateOptions>>,
    last_chat_messages: Mutex<Vec<ChatMessage>>,
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
            stream_mode: MockStreamMode::Normal,
            last_options: Mutex::new(None),
            last_chat_messages: Mutex::new(Vec::new()),
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
        options: GenerateOptions,
    ) -> BoxFuture<'a, Result<mpsc::UnboundedReceiver<GenerateEvent>>> {
        let fail = self.fail_with;
        let prompt_len = prompt.len();
        *self.last_options.lock().unwrap() = Some(options);
        let mode = match self.stream_mode {
            MockStreamMode::Normal => 0,
            MockStreamMode::EmptyDeltaThenFinished => 1,
            MockStreamMode::EmptyDeltaThenError => 2,
            MockStreamMode::MidStreamUnavailableError => 3,
        };
        Box::pin(async move {
            if let Some(make_err) = fail {
                return Err(make_err());
            }
            let (tx, rx) = mpsc::unbounded_channel();
            tokio::spawn(async move {
                match mode {
                    0 => {
                        let _ = tx.send(GenerateEvent::Delta {
                            token_id: 1,
                            text: "hello ".to_string(),
                        });
                        let _ = tx.send(GenerateEvent::Delta {
                            token_id: 2,
                            text: "world".to_string(),
                        });
                        let _ = tx.send(GenerateEvent::Finished {
                            reason: FinishReason::Stop,
                            usage: Usage {
                                prompt_tokens: prompt_len,
                                completion_tokens: 2,
                            },
                        });
                    }
                    1 => {
                        let _ = tx.send(GenerateEvent::Delta {
                            token_id: 1,
                            text: String::new(),
                        });
                        let _ = tx.send(GenerateEvent::Delta {
                            token_id: 2,
                            text: "ok".to_string(),
                        });
                        let _ = tx.send(GenerateEvent::Finished {
                            reason: FinishReason::Stop,
                            usage: Usage {
                                prompt_tokens: prompt_len,
                                completion_tokens: 2,
                            },
                        });
                    }
                    2 => {
                        let _ = tx.send(GenerateEvent::Delta {
                            token_id: 1,
                            text: String::new(),
                        });
                        let _ = tx.send(GenerateEvent::Error(Error::new("gpu boom")));
                    }
                    _ => {
                        let _ = tx.send(GenerateEvent::Error(Error::unavailable("draining")));
                    }
                }
            });
            Ok(rx)
        })
    }

    fn submit_chat<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        options: GenerateOptions,
        enable_thinking: Option<bool>,
    ) -> BoxFuture<'a, Result<mpsc::UnboundedReceiver<GenerateEvent>>> {
        let fail = self.fail_with;
        let count = messages.len();
        *self.last_options.lock().unwrap() = Some(options);
        *self.last_chat_messages.lock().unwrap() = messages.to_vec();
        let mode = match self.stream_mode {
            MockStreamMode::MidStreamUnavailableError => 3,
            _ => 0,
        };
        Box::pin(async move {
            if let Some(make_err) = fail {
                return Err(make_err());
            }
            let (tx, rx) = mpsc::unbounded_channel();
            tokio::spawn(async move {
                if mode == 3 {
                    let _ = tx.send(GenerateEvent::Error(Error::unavailable("draining")));
                    return;
                }
                let tag = if enable_thinking == Some(false) {
                    "no-think"
                } else {
                    "ok"
                };
                let _ = tx.send(GenerateEvent::Delta {
                    token_id: 10,
                    text: format!("{tag}:{count}"),
                });
                let _ = tx.send(GenerateEvent::Finished {
                    reason: FinishReason::Stop,
                    usage: Usage {
                        prompt_tokens: 5,
                        completion_tokens: 1,
                    },
                });
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

async fn post_json(addr: std::net::SocketAddr, path: &str, payload: &str) -> (u16, String) {
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    send_http(addr, &req).await
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

    // Non-streaming /v1/completions + ignore_eos & stop_token_ids forwarding
    let payload = r#"{"model":"Qwen/Mock-0.8B","prompt":"hi","max_tokens":4,"ignore_eos":true,"stop_token_ids":[151643,151645]}"#;
    let (code, body) = post_json(addr, "/v1/completions", payload).await;
    assert_eq!(code, 200);
    let resp: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(resp["choices"][0]["text"], "hello world");
    assert_eq!(resp["choices"][0]["finish_reason"], "stop");
    assert_eq!(resp["usage"]["completion_tokens"], 2);
    {
        let opts = backend.last_options.lock().unwrap();
        let opts = opts.as_ref().unwrap();
        assert!(opts.ignore_eos);
        assert_eq!(opts.stop_token_ids, vec![151643, 151645]);
    }

    // Streaming /v1/chat/completions (SSE) with structured content parts + null assistant content
    let payload = r#"{"model":"Qwen/Mock-0.8B","messages":[{"role":"user","content":[{"type":"text","text":"hi "} ,{"type":"text","text":"there"}]},{"role":"assistant","content":null}],"stream":true,"ignore_eos":true,"stop_token_ids":[42],"chat_template_kwargs":{"enable_thinking":false}}"#;
    let (code, body) = post_json(addr, "/v1/chat/completions", payload).await;
    assert_eq!(code, 200);
    assert!(body.contains(r#""role":"assistant""#));
    assert!(body.contains(r#""content":"no-think:2""#));
    assert!(body.contains(r#""finish_reason":"stop""#));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    {
        let msgs = backend.last_chat_messages.lock().unwrap();
        assert_eq!(msgs[0].content, "hi there");
        assert_eq!(msgs[1].content, "");
        let opts = backend.last_options.lock().unwrap();
        assert!(opts.as_ref().unwrap().ignore_eos);
        assert_eq!(opts.as_ref().unwrap().stop_token_ids, vec![42]);
    }

    // Unknown model -> 404
    let (code, body) = post_json(
        addr,
        "/v1/completions",
        r#"{"model":"wrong-model","prompt":"hi"}"#,
    )
    .await;
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
async fn error_kinds_map_without_substring_matching() {
    // QueueFull -> 429 regardless of message text.
    let mut mock_q = MockBackend::new();
    mock_q.fail_with = Some(|| Error::queue_full("server overloaded"));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, build_router(Arc::new(mock_q)))
            .await
            .unwrap();
    });
    let (code, body) = post_json(addr, "/v1/completions", r#"{"prompt":"hi"}"#).await;
    assert_eq!(code, 429);
    assert!(body.contains("queue_full"));
    server.abort();

    // Unavailable -> 503 regardless of message text.
    let mut mock_u = MockBackend::new();
    mock_u.fail_with = Some(|| Error::unavailable("draining"));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, build_router(Arc::new(mock_u)))
            .await
            .unwrap();
    });
    let (code, body) = post_json(addr, "/v1/completions", r#"{"prompt":"hi"}"#).await;
    assert_eq!(code, 503);
    assert!(body.contains("service_unavailable"));
    server.abort();
}

#[tokio::test]
async fn sse_completions_single_done_and_midstream_errors() {
    // Empty delta followed by real delta + Finished must emit [DONE] exactly once.
    let mut mock_ok = MockBackend::new();
    mock_ok.stream_mode = MockStreamMode::EmptyDeltaThenFinished;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, build_router(Arc::new(mock_ok)))
            .await
            .unwrap();
    });
    let (code, body) = post_json(addr, "/v1/completions", r#"{"prompt":"hi","stream":true}"#).await;
    assert_eq!(code, 200);
    assert!(body.contains(r#""text":"ok""#));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    server.abort();

    // Empty delta followed by Error must emit an SSE error payload and omit [DONE].
    let mut mock_err = MockBackend::new();
    mock_err.stream_mode = MockStreamMode::EmptyDeltaThenError;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, build_router(Arc::new(mock_err)))
            .await
            .unwrap();
    });
    let (code, body) = post_json(addr, "/v1/completions", r#"{"prompt":"hi","stream":true}"#).await;
    assert_eq!(code, 200);
    assert!(body.contains("gpu boom"));
    assert!(body.contains("server_error"));
    assert!(!body.contains("[DONE]"));
    server.abort();

    // Chat streaming mid-stream Error must emit SSE error payload and omit [DONE].
    let mut mock_chat_err = MockBackend::new();
    mock_chat_err.stream_mode = MockStreamMode::MidStreamUnavailableError;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, build_router(Arc::new(mock_chat_err)))
            .await
            .unwrap();
    });
    let (code, body) = post_json(
        addr,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#,
    )
    .await;
    assert_eq!(code, 200);
    assert!(body.contains("service_unavailable"));
    assert!(body.contains("draining"));
    assert!(!body.contains("[DONE]"));
    server.abort();
}
