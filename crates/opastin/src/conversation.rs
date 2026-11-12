//! Cache-aware routing support: remember which model served a conversation, so that the
//! next turn can account for the provider's prompt cache.
//!
//! Providers cache prompt prefixes (Anthropic, OpenAI, Gemini and others bill cached
//! input tokens at a fraction of the normal price). A multi-turn conversation re-sends
//! the whole history every turn, so after a few turns most of the input is cached
//! *at the model that served the previous turn*. Switching models loses that cache and
//! pays full input price for the whole history again.
//!
//! No session id is needed from the client. A request's history is fingerprinted
//! message by message (`h_0` = API format + system prompt + tools, `h_k` = `h_{k-1}` +
//! message k). After a turn is served, the fingerprint of its full message list is
//! stored with the model that served it. The next turn of the same conversation repeats
//! that list as a prefix (plus the assistant reply and the new user message), so the
//! longest stored prefix identifies the previous model and roughly how many prompt
//! tokens its cache holds.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::ApiFormat;
use crate::router::Candidate;

/// Rough prompt-token estimate from serialized JSON size (about 4 bytes per token).
const BYTES_PER_TOKEN: u64 = 4;

/// What is known about the conversation a request continues.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conversation {
    /// Model that served the previous turn.
    pub previous_model: String,
    /// Estimated prompt tokens likely cached at the previous model's provider: system
    /// prompt, tools, and the history up to the previous turn.
    pub cached_tokens: u64,
}

/// Per-model prices in USD per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pricing {
    pub input: f64,
    /// Price of input tokens read from the prompt cache.
    pub cached_input: f64,
    pub output: f64,
}

/// Extra USD a candidate costs because its prompt cache is cold for this conversation:
/// the cached prefix billed at full input price instead of the cached price. Zero for
/// the model that served the previous turn, and when pricing is unknown.
pub fn switch_cost(candidate: &Candidate, conversation: Option<&Conversation>) -> f64 {
    match (conversation, &candidate.pricing) {
        (Some(c), Some(p)) if c.previous_model != candidate.id => {
            c.cached_tokens as f64 * (p.input - p.cached_input).max(0.0) / 1e6
        }
        _ => 0.0,
    }
}

/// Fingerprints of every message prefix of one request.
#[derive(Debug, Clone)]
pub struct PrefixKeys {
    /// `(hash, estimated tokens)` for prefixes of 0..=n messages.
    keys: Vec<(u64, u64)>,
}

impl PrefixKeys {
    pub fn new(format: ApiFormat, body: &Value) -> Self {
        let mut h = DefaultHasher::new();
        format.hash(&mut h);
        let mut tokens = 0;
        // Everything that precedes the messages in the provider's cached prefix.
        for field in ["system", "tools"] {
            if let Some(v) = body.get(field) {
                let s = v.to_string();
                field.hash(&mut h);
                s.hash(&mut h);
                tokens += s.len() as u64 / BYTES_PER_TOKEN;
            }
        }
        let mut keys = vec![(h.finish(), tokens)];
        for message in body.get("messages").and_then(Value::as_array).into_iter().flatten() {
            let s = message.to_string();
            s.hash(&mut h);
            tokens += s.len() as u64 / BYTES_PER_TOKEN;
            keys.push((h.finish(), tokens));
        }
        Self { keys }
    }

    /// Number of messages in the request.
    pub fn len(&self) -> usize {
        self.keys.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated prompt tokens of the whole request (system, tools and messages).
    pub fn total_tokens(&self) -> u64 {
        self.keys.last().map_or(0, |k| k.1)
    }
}

struct Entry {
    model: String,
    at: Instant,
}

/// In-memory map from conversation fingerprints to the model that served them.
pub struct ConversationTracker {
    ttl: Duration,
    max_entries: usize,
    entries: Mutex<HashMap<u64, Entry>>,
}

impl ConversationTracker {
    /// `ttl` should match the providers' cache lifetime (Anthropic's default is 5 min).
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            ttl,
            max_entries: max_entries.max(1),
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// The conversation this request continues, if a strict prefix of its messages was
    /// served within the TTL. The longest matching prefix wins.
    pub fn lookup(&self, keys: &PrefixKeys) -> Option<Conversation> {
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let now = Instant::now();
        (1..keys.keys.len().saturating_sub(1)).rev().find_map(|k| {
            let (hash, tokens) = keys.keys[k];
            entries
                .get(&hash)
                .filter(|e| now.duration_since(e.at) < self.ttl)
                .map(|e| Conversation { previous_model: e.model.clone(), cached_tokens: tokens })
        })
    }

    /// Record that `model` served this request.
    pub fn record(&self, keys: &PrefixKeys, model: &str) {
        if keys.is_empty() {
            return;
        }
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let now = Instant::now();
        if entries.len() >= self.max_entries {
            entries.retain(|_, e| now.duration_since(e.at) < self.ttl);
            if entries.len() >= self.max_entries {
                // Still full: drop the oldest half.
                let mut ages: Vec<Instant> = entries.values().map(|e| e.at).collect();
                ages.sort_unstable();
                let cutoff = ages[ages.len() / 2];
                entries.retain(|_, e| e.at > cutoff);
            }
        }
        let (hash, _) = *keys.keys.last().expect("non-empty");
        entries.insert(hash, Entry { model: model.to_owned(), at: now });
    }

    pub fn len(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn turn1() -> Value {
        json!({"system": "You are a coding agent.", "messages": [
            {"role": "user", "content": "Fix the failing test in parser.rs"}
        ]})
    }

    fn turn2() -> Value {
        json!({"system": "You are a coding agent.", "messages": [
            {"role": "user", "content": "Fix the failing test in parser.rs"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "cargo test"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "1 failed"}]}
        ]})
    }

    const F: ApiFormat = ApiFormat::AnthropicMessages;

    #[test]
    fn next_turn_finds_the_previous_model() {
        let t = ConversationTracker::new(Duration::from_secs(300), 100);
        let k1 = PrefixKeys::new(F, &turn1());
        assert_eq!(t.lookup(&k1), None, "first turn is new");
        t.record(&k1, "opus");

        let k2 = PrefixKeys::new(F, &turn2());
        let conv = t.lookup(&k2).expect("continuation");
        assert_eq!(conv.previous_model, "opus");
        assert_eq!(conv.cached_tokens, k1.total_tokens());
        assert!(conv.cached_tokens > 0 && conv.cached_tokens < k2.total_tokens());
    }

    #[test]
    fn longest_prefix_wins_and_other_conversations_do_not_match() {
        let t = ConversationTracker::new(Duration::from_secs(300), 100);
        let k1 = PrefixKeys::new(F, &turn1());
        t.record(&k1, "opus");
        let k2 = PrefixKeys::new(F, &turn2());
        t.record(&k2, "haiku");
        let mut turn3 = turn2();
        turn3["messages"].as_array_mut().unwrap().push(json!({"role": "assistant", "content": "Fixed."}));
        turn3["messages"].as_array_mut().unwrap().push(json!({"role": "user", "content": "Thanks, now add docs"}));
        assert_eq!(t.lookup(&PrefixKeys::new(F, &turn3)).unwrap().previous_model, "haiku");

        // Different system prompt, or different API format: a different conversation.
        let mut other = turn2();
        other["system"] = json!("You are a poet.");
        assert_eq!(t.lookup(&PrefixKeys::new(F, &other)), None);
        assert_eq!(t.lookup(&PrefixKeys::new(ApiFormat::OpenaiChat, &turn2())), None);
    }

    #[test]
    fn identical_request_is_not_its_own_continuation() {
        let t = ConversationTracker::new(Duration::from_secs(300), 100);
        let k1 = PrefixKeys::new(F, &turn1());
        t.record(&k1, "opus");
        assert_eq!(t.lookup(&k1), None, "a retry of the same request is not a continuation");
    }

    #[test]
    fn entries_expire_and_capacity_is_bounded() {
        let t = ConversationTracker::new(Duration::from_millis(0), 100);
        let k1 = PrefixKeys::new(F, &turn1());
        t.record(&k1, "opus");
        assert_eq!(t.lookup(&PrefixKeys::new(F, &turn2())), None, "expired");

        let t = ConversationTracker::new(Duration::from_secs(300), 4);
        for i in 0..20 {
            let body = json!({"messages": [{"role": "user", "content": format!("q{i}")}]});
            t.record(&PrefixKeys::new(F, &body), "m");
        }
        assert!(t.len() <= 4);
    }

    #[test]
    fn switch_cost_prices_the_lost_cache() {
        let conv = Conversation { previous_model: "opus".into(), cached_tokens: 100_000 };
        let priced = |id: &str, input, cached| Candidate {
            id: id.into(),
            pricing: Some(Pricing { input, cached_input: cached, output: 0.0 }),
        };
        // Staying on the cached model costs nothing extra.
        assert_eq!(switch_cost(&priced("opus", 4.0, 0.4), Some(&conv)), 0.0);
        // Switching pays full input price instead of cached for 100k tokens: 100k × $2.7/M.
        let c = switch_cost(&priced("sonnet", 3.0, 0.3), Some(&conv));
        assert!((c - 0.27).abs() < 1e-9);
        // Unknown pricing or no conversation: no penalty.
        assert_eq!(switch_cost(&Candidate::new("x"), Some(&conv)), 0.0);
        assert_eq!(switch_cost(&priced("sonnet", 3.0, 0.3), None), 0.0);
    }
}
