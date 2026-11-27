//! Iteration-level continuous-batching scheduler with interleaved chunked prefill.

use std::collections::VecDeque;
use std::sync::Arc;

use tokio::sync::mpsc;
use xinfer_runtime::{require_kind, Error, ErrorKind, Result, Session, Tensor};

use crate::config::ResolvedEngineConfig;
use crate::sampler::{sample_batch as gpu_sample_batch, RequestRng, SampleRow};
use crate::tokenizer::{BundleTokenizer, StreamDecoder, StreamStep};
use crate::types::{FinishReason, GenerateEvent, GenerateRequest, Usage};

/// Abstraction over the GPU execution backend so scheduler state transitions
/// can be exhaustively verified in CPU-only unit tests with a mock backend.
pub trait BatchBackend {
    type Logits;

    fn capacity(&self) -> usize;
    fn max_prefill(&self) -> usize;
    fn allocated_batch_size(&self) -> usize;
    fn reset_slot(&mut self, slot: usize) -> Result<()>;
    fn step_slot(&mut self, slot: usize, tokens: &[i64]) -> Result<Self::Logits>;
    fn step_slots_range(&mut self, start_slot: usize, rows: &[&[i64]]) -> Result<Self::Logits>;
    fn sample_batch(
        &mut self,
        logits: &Self::Logits,
        params: &[SampleRow],
        active: &[bool],
    ) -> Result<Vec<i64>>;
}

/// Production `BatchBackend` driving `xinfer_runtime::Session` on the dedicated GPU thread.
pub struct SessionBackend {
    session: Session,
}

impl SessionBackend {
    pub fn new(session: Session) -> Self {
        Self { session }
    }
}

impl BatchBackend for SessionBackend {
    type Logits = Tensor;

    fn capacity(&self) -> usize {
        self.session.spec().capacity as usize
    }

    fn max_prefill(&self) -> usize {
        self.session.spec().max_prefill as usize
    }

    fn allocated_batch_size(&self) -> usize {
        self.session.allocated_batch_size() as usize
    }

    fn reset_slot(&mut self, slot: usize) -> Result<()> {
        self.session.reset_slot(slot)
    }

    fn step_slot(&mut self, slot: usize, tokens: &[i64]) -> Result<Tensor> {
        self.session.step_slot(slot, tokens)
    }

    fn step_slots_range(&mut self, start_slot: usize, rows: &[&[i64]]) -> Result<Tensor> {
        self.session.step_slots_range(start_slot, rows)
    }

    fn sample_batch(
        &mut self,
        logits: &Tensor,
        params: &[SampleRow],
        active: &[bool],
    ) -> Result<Vec<i64>> {
        gpu_sample_batch(logits, params, active)
    }
}

pub struct PendingRequest {
    pub request: GenerateRequest,
    pub tx: mpsc::Sender<GenerateEvent>,
}

enum SlotPhase {
    Prefilling { offset: usize, needs_reset: bool },
    Decoding { next_token: i64 },
}

struct SlotEntry {
    request: GenerateRequest,
    eos_ids: Vec<i64>,
    rng: RequestRng,
    decoder: StreamDecoder,
    phase: SlotPhase,
    generated: usize,
    tx: mpsc::Sender<GenerateEvent>,
}

/// Single-thread continuous-batching scheduler managing `1..=allocated_batch_size` slots.
pub struct Scheduler<B: BatchBackend> {
    backend: B,
    tokenizer: Option<Arc<BundleTokenizer>>,
    default_eos: Vec<i64>,
    config: ResolvedEngineConfig,
    slots: Vec<Option<SlotEntry>>,
    ever_used: Vec<bool>,
    queue: VecDeque<PendingRequest>,
    decodes_since_prefill: usize,
}

impl<B: BatchBackend> Scheduler<B> {
    pub fn new(
        backend: B,
        tokenizer: Option<Arc<BundleTokenizer>>,
        default_eos: Vec<i64>,
        config: ResolvedEngineConfig,
    ) -> Self {
        let num_slots = backend.allocated_batch_size();
        Self {
            backend,
            tokenizer,
            default_eos,
            config,
            slots: (0..num_slots).map(|_| None).collect(),
            ever_used: vec![false; num_slots],
            queue: VecDeque::new(),
            decodes_since_prefill: 0,
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn has_work(&self) -> bool {
        !self.queue.is_empty() || self.slots.iter().any(Option::is_some)
    }

    pub fn active_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    pub fn queued_requests(&self) -> usize {
        self.queue.len()
    }

    pub fn submit(&mut self, pending: PendingRequest) -> Result<()> {
        let req = &pending.request;
        req.sampling.validate()?;
        require_kind(
            !req.prompt_tokens.is_empty(),
            ErrorKind::InvalidInput,
            "Prompt must contain at least 1 token",
        )?;
        require_kind(
            req.max_tokens >= 1,
            ErrorKind::InvalidInput,
            "max_tokens must be >= 1",
        )?;
        let capacity = self.backend.capacity();
        let total = req.prompt_tokens.len().saturating_add(req.max_tokens);
        require_kind(
            total <= capacity,
            ErrorKind::CapacityExceeded,
            format!(
                "Prompt ({}) + max_tokens ({}) exceeds bundle context capacity ({})",
                req.prompt_tokens.len(),
                req.max_tokens,
                capacity
            ),
        )?;
        require_kind(
            self.queue.len() < self.config.max_queued_requests,
            ErrorKind::CapacityExceeded,
            "Request queue is full",
        )?;

        self.queue.push_back(pending);
        Ok(())
    }

    /// Aborts all queued and active requests (used during shutdown or unrecoverable GPU errors).
    pub fn abort_all(&mut self, message: &str) {
        while let Some(pending) = self.queue.pop_front() {
            let _ = pending
                .tx
                .try_send(GenerateEvent::Error(Error::new(message)));
        }
        for slot in &mut self.slots {
            if let Some(entry) = slot.take() {
                let _ = entry.tx.try_send(GenerateEvent::Error(Error::new(message)));
            }
        }
    }

    /// Runs one scheduler iteration: reaps cancelled streams, admits queued requests into the
    /// lowest free slots, and executes either one chunked prefill or one batched decode step.
    pub fn step(&mut self) -> Result<bool> {
        self.reap_cancelled();
        self.admit_waiting();

        let prefill_slot = self.slots.iter().position(|slot| {
            matches!(
                slot,
                Some(SlotEntry {
                    phase: SlotPhase::Prefilling { .. },
                    ..
                })
            )
        });
        let has_decode = self.slots.iter().any(|slot| {
            matches!(
                slot,
                Some(SlotEntry {
                    phase: SlotPhase::Decoding { .. },
                    ..
                })
            )
        });

        let run_prefill = match (prefill_slot, has_decode) {
            (Some(_), false) => true,
            (Some(_), true) => {
                self.decodes_since_prefill >= self.config.decode_steps_per_prefill_chunk
            }
            (None, true) => false,
            (None, false) => return Ok(false),
        };

        if run_prefill {
            self.decodes_since_prefill = 0;
            self.run_prefill_chunk(prefill_slot.unwrap())?;
        } else {
            self.decodes_since_prefill += 1;
            self.run_decode_batch()?;
        }
        Ok(true)
    }

    fn reap_cancelled(&mut self) {
        self.queue.retain(|pending| !pending.tx.is_closed());
        for slot in &mut self.slots {
            if slot.as_ref().is_some_and(|entry| entry.tx.is_closed()) {
                *slot = None;
            }
        }
    }

    fn admit_waiting(&mut self) {
        while !self.queue.is_empty() {
            let Some(free_slot) = self.slots.iter().position(Option::is_none) else {
                break;
            };
            let pending = self.queue.pop_front().unwrap();
            if pending.tx.is_closed() {
                continue;
            }
            let needs_reset = self.ever_used[free_slot];
            self.ever_used[free_slot] = true;

            let mut eos_ids = if pending.request.ignore_eos {
                Vec::new()
            } else {
                self.default_eos.clone()
            };
            for &id in &pending.request.stop_token_ids {
                if !eos_ids.contains(&id) {
                    eos_ids.push(id);
                }
            }

            let rng = RequestRng::from_seed(pending.request.sampling.seed);
            let decoder =
                StreamDecoder::new(self.tokenizer.clone(), pending.request.stop_strings.clone());
            self.slots[free_slot] = Some(SlotEntry {
                request: pending.request,
                eos_ids,
                rng,
                decoder,
                phase: SlotPhase::Prefilling {
                    offset: 0,
                    needs_reset,
                },
                generated: 0,
                tx: pending.tx,
            });
        }
    }

    fn run_prefill_chunk(&mut self, slot: usize) -> Result<()> {
        let (chunk, needs_reset, is_final) = {
            let entry = self.slots[slot].as_ref().unwrap();
            let SlotPhase::Prefilling {
                offset,
                needs_reset,
            } = entry.phase
            else {
                unreachable!();
            };
            let chunk_size = self
                .config
                .prefill_chunk_size
                .min(self.backend.max_prefill());
            let end = (offset + chunk_size).min(entry.request.prompt_tokens.len());
            (
                entry.request.prompt_tokens[offset..end].to_vec(),
                needs_reset,
                end == entry.request.prompt_tokens.len(),
            )
        };

        if needs_reset {
            self.backend.reset_slot(slot)?;
        }

        let logits = self.backend.step_slot(slot, &chunk)?;

        if !is_final {
            let entry = self.slots[slot].as_mut().unwrap();
            if let SlotPhase::Prefilling {
                offset,
                needs_reset: flag,
            } = &mut entry.phase
            {
                *offset += chunk.len();
                *flag = false;
            }
            return Ok(());
        }

        let sample_row = {
            let entry = self.slots[slot].as_mut().unwrap();
            SampleRow::from_params(&entry.request.sampling, &mut entry.rng)
        };
        let sampled = self.backend.sample_batch(&logits, &[sample_row], &[true])?;
        self.consume_sampled_token(slot, sampled[0])?;
        Ok(())
    }

    fn run_decode_batch(&mut self) -> Result<()> {
        let mut first_slot: Option<usize> = None;
        let mut last_slot: usize = 0;
        for (idx, slot) in self.slots.iter().enumerate() {
            if matches!(
                slot,
                Some(SlotEntry {
                    phase: SlotPhase::Decoding { .. },
                    ..
                })
            ) {
                if first_slot.is_none() {
                    first_slot = Some(idx);
                }
                last_slot = idx;
            }
        }
        let Some(first) = first_slot else {
            return Ok(());
        };
        let span = last_slot - first + 1;

        let mut token_storage: Vec<[i64; 1]> = vec![[0]; span];
        let mut active: Vec<bool> = vec![false; span];
        let mut sample_rows: Vec<SampleRow> = vec![
            SampleRow {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                uniform: 0.0,
            };
            span
        ];

        for offset in 0..span {
            let slot_idx = first + offset;
            if let Some(entry) = self.slots[slot_idx].as_mut() {
                if let SlotPhase::Decoding { next_token } = entry.phase {
                    token_storage[offset][0] = next_token;
                    active[offset] = true;
                    sample_rows[offset] =
                        SampleRow::from_params(&entry.request.sampling, &mut entry.rng);
                }
            }
        }

        let rows: Vec<&[i64]> = (0..span)
            .map(|offset| {
                if active[offset] {
                    &token_storage[offset][..]
                } else {
                    &[][..]
                }
            })
            .collect();

        let logits = self.backend.step_slots_range(first, &rows)?;
        let sampled = self.backend.sample_batch(&logits, &sample_rows, &active)?;

        for offset in 0..span {
            if active[offset] {
                self.consume_sampled_token(first + offset, sampled[offset])?;
            }
        }
        Ok(())
    }

    fn consume_sampled_token(&mut self, slot: usize, token_id: i64) -> Result<()> {
        let mut retire = false;
        let entry = self.slots[slot].as_mut().unwrap();
        entry.generated += 1;

        let hit_eos = entry.eos_ids.contains(&token_id);
        let hit_limit = entry.generated >= entry.request.max_tokens;
        let usage = Usage {
            prompt_tokens: entry.request.prompt_tokens.len(),
            completion_tokens: entry.generated,
        };

        if hit_eos {
            let tail = entry.decoder.finish()?;
            if !tail.is_empty() {
                let _ = entry.tx.blocking_send(GenerateEvent::Delta {
                    token_id,
                    text: tail,
                });
            }
            let _ = entry.tx.blocking_send(GenerateEvent::Finished {
                reason: FinishReason::Stop,
                usage,
            });
            retire = true;
        } else {
            match entry.decoder.push(token_id)? {
                StreamStep::Stopped(text) => {
                    if !text.is_empty() {
                        let _ = entry
                            .tx
                            .blocking_send(GenerateEvent::Delta { token_id, text });
                    }
                    let _ = entry.tx.blocking_send(GenerateEvent::Finished {
                        reason: FinishReason::Stop,
                        usage,
                    });
                    retire = true;
                }
                StreamStep::Continue(mut text) => {
                    if hit_limit {
                        let tail = entry.decoder.finish()?;
                        text.push_str(&tail);
                        let _ = entry
                            .tx
                            .blocking_send(GenerateEvent::Delta { token_id, text });
                        let _ = entry.tx.blocking_send(GenerateEvent::Finished {
                            reason: FinishReason::Length,
                            usage,
                        });
                        retire = true;
                    } else if entry
                        .tx
                        .blocking_send(GenerateEvent::Delta { token_id, text })
                        .is_err()
                    {
                        retire = true;
                    } else {
                        entry.phase = SlotPhase::Decoding {
                            next_token: token_id,
                        };
                    }
                }
            }
        }

        if retire {
            self.slots[slot] = None;
        }
        Ok(())
    }
}
