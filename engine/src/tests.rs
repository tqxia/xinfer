use tokio::sync::mpsc;
use xinfer_runtime::{DType, Device, ErrorKind, Result, Tensor};

use crate::{
    sample_batch, BatchBackend, FinishReason, GenerateEvent, GenerateRequest, PendingRequest,
    RequestRng, ResolvedEngineConfig, SampleRow, SamplingParams, Scheduler, StreamDecoder,
    StreamStep,
};

#[derive(Debug, PartialEq, Eq)]
enum CallRecord {
    ResetSlot(usize),
    StepSlot(usize, Vec<i64>),
    StepRange(usize, Vec<Vec<i64>>),
}

/// Deterministic mock backend verifying slot isolation, chunked prefill, interior holes,
/// recycling, and interleaved cadence without GPU or LibTorch.
struct MockBackend {
    capacity: usize,
    max_prefill: usize,
    allocated_batch_size: usize,
    positions: Vec<usize>,
    last_token: Vec<i64>,
    calls: Vec<CallRecord>,
}

impl MockBackend {
    fn new(allocated_batch_size: usize, capacity: usize, max_prefill: usize) -> Self {
        Self {
            capacity,
            max_prefill,
            allocated_batch_size,
            positions: vec![0; allocated_batch_size],
            last_token: vec![0; allocated_batch_size],
            calls: Vec::new(),
        }
    }
}

impl BatchBackend for MockBackend {
    type Logits = Vec<i64>;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn max_prefill(&self) -> usize {
        self.max_prefill
    }

    fn allocated_batch_size(&self) -> usize {
        self.allocated_batch_size
    }

    fn reset_slot(&mut self, slot: usize) -> Result<()> {
        self.calls.push(CallRecord::ResetSlot(slot));
        self.positions[slot] = 0;
        self.last_token[slot] = 0;
        Ok(())
    }

    fn step_slot(&mut self, slot: usize, tokens: &[i64]) -> Result<Vec<i64>> {
        self.calls.push(CallRecord::StepSlot(slot, tokens.to_vec()));
        self.positions[slot] += tokens.len();
        self.last_token[slot] = *tokens.last().unwrap();
        Ok(vec![self.last_token[slot] + 1])
    }

    fn step_slots_range(&mut self, start_slot: usize, rows: &[&[i64]]) -> Result<Vec<i64>> {
        self.calls.push(CallRecord::StepRange(
            start_slot,
            rows.iter().map(|r| r.to_vec()).collect(),
        ));
        let mut out = Vec::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            let slot = start_slot + i;
            if row.is_empty() {
                out.push(0);
            } else {
                self.positions[slot] += row.len();
                self.last_token[slot] = *row.last().unwrap();
                out.push(self.last_token[slot] + 1);
            }
        }
        Ok(out)
    }

    fn sample_batch(
        &mut self,
        logits: &Vec<i64>,
        _params: &[SampleRow],
        active: &[bool],
    ) -> Result<Vec<i64>> {
        Ok(logits
            .iter()
            .zip(active.iter())
            .map(|(&tok, &is_active)| if is_active { tok } else { 0 })
            .collect())
    }
}

fn test_config(batch: usize, chunk: usize, decode_per_prefill: usize) -> ResolvedEngineConfig {
    ResolvedEngineConfig {
        allocated_batch_size: batch,
        prefill_chunk_size: chunk,
        decode_steps_per_prefill_chunk: decode_per_prefill,
        max_queued_requests: 4,
        served_model_name: "mock".to_string(),
    }
}

fn make_req(
    prompt: Vec<i64>,
    max_tokens: usize,
) -> (PendingRequest, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel(32);
    let pending = PendingRequest {
        request: GenerateRequest {
            prompt_tokens: prompt,
            max_tokens,
            sampling: SamplingParams::default(),
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            ignore_eos: false,
        },
        tx,
    };
    (pending, rx)
}

fn collect_events(mut rx: mpsc::Receiver<GenerateEvent>) -> (Vec<i64>, FinishReason) {
    let mut ids = Vec::new();
    let mut reason = None;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            GenerateEvent::Delta { token_id, .. } => ids.push(token_id),
            GenerateEvent::Finished { reason: r, .. } => reason = Some(r),
            GenerateEvent::Error(err) => panic!("unexpected error event: {err}"),
        }
    }
    (ids, reason.expect("expected Finished event"))
}

#[test]
fn interleaved_chunked_prefill_and_decode_progress() {
    // Chunk size 2, 1 decode step per prefill chunk.
    let backend = MockBackend::new(2, 64, 8);
    let mut sched = Scheduler::new(backend, None, vec![999], test_config(2, 2, 1));

    // Request A (2 tokens -> 1 chunk, 3 output tokens) starts decoding on slot 0.
    let (req_a, rx_a) = make_req(vec![10, 11], 3);
    sched.submit(req_a).unwrap();
    sched.step().unwrap(); // Prefill slot 0 -> emits 12, moves to Decoding.

    // Now Request B (5 tokens -> 3 chunks: [2, 2, 1]) arrives for slot 1.
    let (req_b, rx_b) = make_req(vec![100, 101, 102, 103, 104], 2);
    sched.submit(req_b).unwrap();

    while sched.has_work() {
        sched.step().unwrap();
    }

    let (ids_a, reason_a) = collect_events(rx_a);
    let (ids_b, reason_b) = collect_events(rx_b);
    assert_eq!(ids_a, vec![12, 13, 14]);
    assert_eq!(reason_a, FinishReason::Length);
    assert_eq!(ids_b, vec![105, 106]);
    assert_eq!(reason_b, FinishReason::Length);

    // Verify slot 0 continued decoding interleaved between slot 1's prefill chunks!
    let calls = &sched.backend().calls;
    assert_eq!(calls[0], CallRecord::StepSlot(0, vec![10, 11]));
    assert_eq!(calls[1], CallRecord::StepRange(0, vec![vec![12]]));
    assert_eq!(calls[2], CallRecord::StepSlot(1, vec![100, 101]));
    assert_eq!(calls[3], CallRecord::StepRange(0, vec![vec![13]]));
    assert_eq!(calls[4], CallRecord::StepSlot(1, vec![102, 103]));
    // Slot 0 finished after producing 14; slot 1 finishes its final singleton chunk and decodes.
    assert_eq!(calls[5], CallRecord::StepSlot(1, vec![104]));
    assert_eq!(calls[6], CallRecord::StepRange(1, vec![vec![105]]));
}

#[test]
fn slot_recycling_resets_slot_and_handles_interior_holes_and_cancellation() {
    let backend = MockBackend::new(3, 64, 8);
    let mut sched = Scheduler::new(backend, None, vec![50], test_config(3, 4, 1));

    // Fill all 3 slots: slot 0, 1, 2.
    let (r0, rx0) = make_req(vec![10, 11], 6);
    let (r1, rx1) = make_req(vec![20, 21], 6);
    let (r2, rx2) = make_req(vec![30, 31], 6);
    sched.submit(r0).unwrap();
    sched.submit(r1).unwrap();
    sched.submit(r2).unwrap();

    // Run 5 interleaved steps (prefill 0, decode 0, prefill 1, decode 0..1, prefill 2) -> all 3 decoding.
    for _ in 0..5 {
        sched.step().unwrap();
    }

    // Client 1 disconnects mid-stream -> slot 1 becomes an interior hole between slot 0 and slot 2.
    drop(rx1);
    sched.step().unwrap();
    assert!(sched.backend().calls.iter().any(
        |c| matches!(c, CallRecord::StepRange(0, rows) if rows.len() == 3 && rows[1].is_empty())
    ));

    // Submit Request 3 -> must reuse lowest free slot (slot 1) with lazy reset_slot(1).
    let (r3, rx3) = make_req(vec![48, 49], 5); // 49 + 1 = 50 == EOS!
    sched.submit(r3).unwrap();
    while sched.has_work() {
        sched.step().unwrap();
    }

    assert!(sched.backend().calls.contains(&CallRecord::ResetSlot(1)));
    let (ids0, reason0) = collect_events(rx0);
    let (ids2, reason2) = collect_events(rx2);
    let (ids3, reason3) = collect_events(rx3);
    assert_eq!(ids0, vec![12, 13, 14, 15, 16, 17]);
    assert_eq!(reason0, FinishReason::Length);
    assert_eq!(ids2, vec![32, 33, 34, 35, 36, 37]);
    assert_eq!(reason2, FinishReason::Length);
    // Request 3 hit EOS on first generated token (50).
    assert_eq!(ids3, Vec::<i64>::new());
    assert_eq!(reason3, FinishReason::Stop);
}

#[test]
fn scheduler_enforces_capacity_and_queue_limits() {
    let backend = MockBackend::new(1, 10, 4);
    let mut sched = Scheduler::new(backend, None, vec![99], test_config(1, 4, 1));

    let (bad_cap, _rx) = make_req(vec![1; 8], 3);
    let err = sched.submit(bad_cap).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::CapacityExceeded);

    let (empty, _rx) = make_req(vec![], 1);
    assert_eq!(
        sched.submit(empty).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );

    // Queue bound is 4.
    let mut rxs = Vec::new();
    for _ in 0..4 {
        let (req, rx) = make_req(vec![1, 2], 1);
        sched.submit(req).unwrap();
        rxs.push(rx);
    }
    let (overflow, _rx) = make_req(vec![1, 2], 1);
    let err = sched.submit(overflow).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::CapacityExceeded);
    assert!(err.to_string().contains("queue"));
}

#[test]
fn stream_decoder_holds_partial_stop_prefixes_and_truncates_match() {
    let mut dec = StreamDecoder::new(None, vec!["STOP".to_string(), "\n\nUser:".to_string()]);
    assert_eq!(
        dec.push_text_for_test("Hello ST").unwrap(),
        StreamStep::Continue("Hello ".to_string())
    );
    // Partial prefix "ST" followed by "a" is not a stop string -> flushes "STa".
    assert_eq!(
        dec.push_text_for_test("art").unwrap(),
        StreamStep::Continue("STart".to_string())
    );
    // Split stop string across chunks: "ok ST" -> emits "ok ", holds "ST"; "OP!!!" -> stops with "".
    assert_eq!(
        dec.push_text_for_test(" ok ST").unwrap(),
        StreamStep::Continue(" ok ".to_string())
    );
    assert_eq!(
        dec.push_text_for_test("OP!!!").unwrap(),
        StreamStep::Stopped(String::new())
    );

    // Pending prefix flushes on finish() if EOS arrives before stop string completes.
    let mut dec2 = StreamDecoder::new(None, vec!["END".to_string()]);
    assert_eq!(
        dec2.push_text_for_test("hi EN").unwrap(),
        StreamStep::Continue("hi ".to_string())
    );
    assert_eq!(dec2.finish().unwrap(), "EN");
}

#[test]
fn sampler_shim_greedy_topk_topp_and_seed_determinism_on_cpu() {
    let logits = Tensor::empty_contiguous(&[3, 5], DType::Float32, Device::Cpu).unwrap();
    let ptr = logits.data_ptr().unwrap().cast::<f32>();
    let values: [f32; 15] = [
        -1.0, 10.0, 0.0, 1.0, -3.0, // row 0: argmax=1
        0.0, 0.0, 5.0, 4.9, -10.0, // row 1: top2 = {2, 3}
        1.0, 2.0, 3.0, 4.0, 5.0, // row 2: inactive
    ];
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }

    // Greedy path with inactive row 2.
    let greedy = sample_batch(
        &logits,
        &[
            SampleRow {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                uniform: 0.0,
            },
            SampleRow {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                uniform: 0.0,
            },
            SampleRow {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                uniform: 0.0,
            },
        ],
        &[true, true, false],
    )
    .unwrap();
    assert_eq!(greedy, vec![1, 2, 0]);

    // Stochastic top_k=2 + deterministic seed must reproduce identical picks and stay in {2, 3}.
    let mut rng_a = RequestRng::from_seed(Some(42));
    let mut rng_b = RequestRng::from_seed(Some(42));
    let u_a = rng_a.next_uniform_f32();
    let u_b = rng_b.next_uniform_f32();
    assert_eq!(u_a, u_b);

    let pick_low = sample_batch(
        &logits,
        &[
            SampleRow {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                uniform: 0.0,
            },
            SampleRow {
                temperature: 1.0,
                top_p: 0.99,
                top_k: 2,
                uniform: 0.01,
            },
            SampleRow {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                uniform: 0.5,
            },
        ],
        &[true, true, false],
    )
    .unwrap();
    assert_eq!(pick_low, vec![1, 2, 0]);

    let pick_high = sample_batch(
        &logits,
        &[
            SampleRow {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                uniform: 0.0,
            },
            SampleRow {
                temperature: 1.0,
                top_p: 0.99,
                top_k: 2,
                uniform: 0.95,
            },
            SampleRow {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                uniform: 0.5,
            },
        ],
        &[true, true, false],
    )
    .unwrap();
    assert_eq!(pick_high, vec![1, 3, 0]);

    // Pure top_p (no top_k) with very small top_p=0.1 always selects top-1 token (index 2).
    let pick_nucleus = sample_batch(
        &logits,
        &[
            SampleRow {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                uniform: 0.0,
            },
            SampleRow {
                temperature: 1.0,
                top_p: 0.1,
                top_k: 0,
                uniform: 0.99,
            },
            SampleRow {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                uniform: 0.0,
            },
        ],
        &[true, true, false],
    )
    .unwrap();
    assert_eq!(pick_nucleus, vec![1, 2, 0]);
}

#[test]
#[ignore = "requires XINFER_PACKAGES and XINFER_TOK_REF"]
fn bundle_tokenizer_matches_python_transformers_and_streams_utf8_cleanly() {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;

    use crate::{BundleTokenizer, ChatMessage};

    let bundle_dir = PathBuf::from(
        std::env::var("XINFER_PACKAGES").expect("Set XINFER_PACKAGES to a compiled bundle"),
    );
    let ref_path = PathBuf::from(
        std::env::var("XINFER_TOK_REF").expect("Set XINFER_TOK_REF to Python tokenizer reference"),
    );
    let reference: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&ref_path).unwrap()).unwrap();
    let spec = xinfer_runtime::Spec::load(&bundle_dir).unwrap();
    let tok = Arc::new(BundleTokenizer::load(&bundle_dir, &spec.eos).unwrap());

    for item in reference["prompts"].as_array().unwrap() {
        let text = item["text"].as_str().unwrap();
        let expected_ids: Vec<i64> = item["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let expected_decoded = item["decoded"].as_str().unwrap();

        let got_ids = tok.encode(text, true).unwrap();
        assert_eq!(got_ids, expected_ids, "encode mismatch for {text:?}");
        assert_eq!(
            tok.decode(&got_ids, true).unwrap(),
            expected_decoded,
            "decode mismatch for {text:?}"
        );

        let mut decoder = StreamDecoder::new(Some(Arc::clone(&tok)), Vec::new());
        let mut streamed = String::new();
        for &id in &got_ids {
            match decoder.push(id).unwrap() {
                StreamStep::Continue(chunk) => {
                    assert!(
                        !chunk.contains('\u{FFFD}'),
                        "U+FFFD leaked in incremental chunk {chunk:?}"
                    );
                    streamed.push_str(&chunk);
                }
                StreamStep::Stopped(_) => unreachable!(),
            }
        }
        streamed.push_str(&decoder.finish().unwrap());
        assert_eq!(
            streamed, expected_decoded,
            "StreamDecoder mismatch for {text:?}"
        );
    }

    for item in reference["chats"].as_array().unwrap() {
        let messages: Vec<ChatMessage> = serde_json::from_value(item["messages"].clone()).unwrap();
        let enable_thinking = item["enable_thinking"].as_bool();
        let expected_rendered = item["rendered"].as_str().unwrap();
        let expected_ids: Vec<i64> = item["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();

        let rendered = tok
            .apply_chat_template(&messages, true, enable_thinking)
            .unwrap();
        assert_eq!(
            rendered, expected_rendered,
            "chat template mismatch (enable_thinking={enable_thinking:?})"
        );
        let ids = tok.encode(&rendered, false).unwrap();
        assert_eq!(
            ids, expected_ids,
            "chat encoded IDs mismatch (enable_thinking={enable_thinking:?})"
        );
    }
}
