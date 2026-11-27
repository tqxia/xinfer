//! Native Rust tokenizer, Jinja chat template renderer, and incremental UTF-8 / stop-string decoder.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use minijinja::{
    Environment, Error as JinjaError, ErrorKind as JinjaErrorKind, Value as JinjaValue,
};
use serde_json::Value;
use tokenizers::Tokenizer;
use xinfer_runtime::{require_kind, Error, ErrorKind, Result};

use crate::types::ChatMessage;

/// Loaded bundle tokenizer + compiled Jinja chat template + resolved default EOS IDs.
pub struct BundleTokenizer {
    tokenizer: Tokenizer,
    env: Option<Environment<'static>>,
    eos_token_ids: Vec<i64>,
    eos_token_str: Option<String>,
    bos_token_str: Option<String>,
}

fn extract_token_string(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) => Some(s.clone()),
        Value::Object(obj) => obj.get("content").and_then(Value::as_str).map(String::from),
        _ => None,
    }
}

fn compile_chat_template(template_source: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, JinjaError> {
            Err(JinjaError::new(JinjaErrorKind::InvalidOperation, message))
        },
    );
    env.add_template_owned("chat".to_string(), template_source)
        .map_err(|err| Error::unsupported(format!("Failed to compile chat template: {err}")))?;
    Ok(env)
}

impl BundleTokenizer {
    /// Loads `tokenizer.json`, `tokenizer_config.json`, and optional `chat_template.jinja`
    /// from a verified bundle directory.
    pub fn load(bundle_dir: &Path, spec_eos_ids: &[i64]) -> Result<Self> {
        let tok_path = bundle_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|err| {
            Error::io(format!(
                "Failed to load tokenizer {}: {err}",
                tok_path.display()
            ))
        })?;

        let cfg_path = bundle_dir.join("tokenizer_config.json");
        let cfg_text = fs::read_to_string(&cfg_path).map_err(|err| {
            Error::io(format!(
                "Failed to read tokenizer config {}: {err}",
                cfg_path.display()
            ))
        })?;
        let cfg: Value = serde_json::from_str(&cfg_text)?;

        let eos_token_str = extract_token_string(cfg.get("eos_token"));
        let bos_token_str = extract_token_string(cfg.get("bos_token"));

        let mut eos_token_ids = spec_eos_ids.to_vec();
        if let Some(eos_str) = &eos_token_str {
            if let Some(id) = tokenizer.token_to_id(eos_str) {
                let id_i64 = i64::from(id);
                if !eos_token_ids.contains(&id_i64) {
                    eos_token_ids.push(id_i64);
                }
            }
        }

        let jinja_path = bundle_dir.join("chat_template.jinja");
        let template_source = if jinja_path.is_file() {
            Some(fs::read_to_string(&jinja_path).map_err(|err| {
                Error::io(format!(
                    "Failed to read chat template {}: {err}",
                    jinja_path.display()
                ))
            })?)
        } else {
            cfg.get("chat_template")
                .and_then(Value::as_str)
                .map(String::from)
        };

        let env = match template_source {
            Some(source) => Some(compile_chat_template(source)?),
            None => None,
        };

        Ok(Self {
            tokenizer,
            env,
            eos_token_ids,
            eos_token_str,
            bos_token_str,
        })
    }

    pub fn eos_token_ids(&self) -> &[i64] {
        &self.eos_token_ids
    }

    pub fn has_chat_template(&self) -> bool {
        self.env.is_some()
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|err| Error::invalid_input(format!("Tokenization failed: {err}")))?;
        Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    pub fn decode(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String> {
        let u32_ids: Vec<u32> = ids
            .iter()
            .map(|&id| {
                u32::try_from(id)
                    .map_err(|_| Error::invalid_input(format!("Invalid token ID: {id}")))
            })
            .collect::<Result<_>>()?;
        self.tokenizer
            .decode(&u32_ids, skip_special_tokens)
            .map_err(|err| Error::new(format!("Detokenization failed: {err}")))
    }

    /// Renders `messages` through the bundled Jinja chat template.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        require_kind(
            !messages.is_empty(),
            ErrorKind::InvalidInput,
            "messages must not be empty",
        )?;
        let env = self
            .env
            .as_ref()
            .ok_or_else(|| Error::unsupported("Bundle does not include a chat template"))?;
        let template = env
            .get_template("chat")
            .map_err(|err| Error::new(format!("Missing compiled chat template: {err}")))?;

        let mut ctx = BTreeMap::<String, JinjaValue>::new();
        let msg_values: Vec<JinjaValue> = messages
            .iter()
            .map(|m| {
                let mut map = BTreeMap::<String, JinjaValue>::new();
                map.insert("role".to_string(), JinjaValue::from(m.role.as_str()));
                map.insert("content".to_string(), JinjaValue::from(m.content.as_str()));
                JinjaValue::from(map)
            })
            .collect();
        ctx.insert("messages".to_string(), JinjaValue::from(msg_values));
        ctx.insert(
            "add_generation_prompt".to_string(),
            JinjaValue::from(add_generation_prompt),
        );
        ctx.insert(
            "eos_token".to_string(),
            match &self.eos_token_str {
                Some(s) => JinjaValue::from(s.as_str()),
                None => JinjaValue::from(()),
            },
        );
        ctx.insert(
            "bos_token".to_string(),
            match &self.bos_token_str {
                Some(s) => JinjaValue::from(s.as_str()),
                None => JinjaValue::from(()),
            },
        );
        if let Some(thinking) = enable_thinking {
            ctx.insert("enable_thinking".to_string(), JinjaValue::from(thinking));
        }

        template
            .render(JinjaValue::from_object(ctx))
            .map_err(|err| Error::invalid_input(format!("Chat template rendering failed: {err}")))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamStep {
    Continue(String),
    Stopped(String),
}

/// Incremental UTF-8-safe detokenizer with stop-string holdback buffering.
pub struct StreamDecoder {
    tokenizer: Option<Arc<BundleTokenizer>>,
    tokens: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
    stop_strings: Vec<String>,
    pending: String,
}

impl StreamDecoder {
    pub fn new(tokenizer: Option<Arc<BundleTokenizer>>, stop_strings: Vec<String>) -> Self {
        let stop_strings = stop_strings.into_iter().filter(|s| !s.is_empty()).collect();
        Self {
            tokenizer,
            tokens: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
            stop_strings,
            pending: String::new(),
        }
    }

    pub fn push(&mut self, token_id: i64) -> Result<StreamStep> {
        let Some(tok) = &self.tokenizer else {
            return Ok(StreamStep::Continue(String::new()));
        };
        let id = u32::try_from(token_id)
            .map_err(|_| Error::invalid_input(format!("Invalid token ID: {token_id}")))?;
        self.tokens.push(id);

        let prefix_text = tok
            .tokenizer
            .decode(&self.tokens[self.prefix_offset..self.read_offset], true)
            .map_err(|err| Error::new(format!("Detokenization failed: {err}")))?;
        let new_text = tok
            .tokenizer
            .decode(&self.tokens[self.prefix_offset..], true)
            .map_err(|err| Error::new(format!("Detokenization failed: {err}")))?;

        let delta = if new_text.len() > prefix_text.len()
            && !new_text.ends_with('\u{FFFD}')
            && new_text.starts_with(&prefix_text)
        {
            let slice = new_text[prefix_text.len()..].to_string();
            self.prefix_offset = self.read_offset;
            self.read_offset = self.tokens.len();
            slice
        } else {
            String::new()
        };

        self.apply_stop_filter(&delta)
    }

    /// Feeds already-decoded text directly (used by unit tests without a tokenizer file).
    #[cfg(test)]
    pub(crate) fn push_text_for_test(&mut self, delta: &str) -> Result<StreamStep> {
        self.apply_stop_filter(delta)
    }

    fn apply_stop_filter(&mut self, delta: &str) -> Result<StreamStep> {
        if self.stop_strings.is_empty() {
            return Ok(StreamStep::Continue(delta.to_string()));
        }

        self.pending.push_str(delta);
        if let Some(pos) = earliest_stop_match(&self.pending, &self.stop_strings) {
            let emitted = self.pending[..pos].to_string();
            self.pending.clear();
            return Ok(StreamStep::Stopped(emitted));
        }

        let holdback = longest_stop_prefix_suffix(&self.pending, &self.stop_strings);
        let emit_len = self.pending.len().saturating_sub(holdback);
        if emit_len == 0 {
            return Ok(StreamStep::Continue(String::new()));
        }
        let emitted = self.pending[..emit_len].to_string();
        self.pending.drain(..emit_len);
        Ok(StreamStep::Continue(emitted))
    }

    /// Flushes any trailing buffered text (e.g., pending stop holdback or incomplete UTF-8) at EOS/limit.
    pub fn finish(&mut self) -> Result<String> {
        if let Some(tok) = &self.tokenizer {
            if self.read_offset < self.tokens.len() {
                let prefix_text = tok
                    .tokenizer
                    .decode(&self.tokens[self.prefix_offset..self.read_offset], true)
                    .map_err(|err| Error::new(format!("Detokenization failed: {err}")))?;
                let new_text = tok
                    .tokenizer
                    .decode(&self.tokens[self.prefix_offset..], true)
                    .map_err(|err| Error::new(format!("Detokenization failed: {err}")))?;
                if new_text.len() > prefix_text.len() && new_text.starts_with(&prefix_text) {
                    self.pending.push_str(&new_text[prefix_text.len()..]);
                }
                self.read_offset = self.tokens.len();
            }
        }
        Ok(std::mem::take(&mut self.pending))
    }
}

fn earliest_stop_match(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter_map(|stop| text.find(stop.as_str()))
        .min()
}

/// Returns the byte length of the longest suffix of `text` that is a proper prefix
/// of any string in `stops` (always on a UTF-8 char boundary).
fn longest_stop_prefix_suffix(text: &str, stops: &[String]) -> usize {
    let mut best = 0usize;
    for (byte_idx, _) in text.char_indices() {
        let suffix = &text[byte_idx..];
        if stops.iter().any(|stop| stop.starts_with(suffix)) {
            best = best.max(suffix.len());
        }
    }
    best
}
