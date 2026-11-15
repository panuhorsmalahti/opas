//! Recent routing decisions, each with what was predicted and, once the response has
//! finished, what actually happened: tokens, cost and timing.
//!
//! Kept in memory (the most recent [`DecisionLog::DEFAULT_CAPACITY`] decisions) and
//! served by `GET /v1/decisions/{id}`.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::api::ApiFormat;
use crate::conversation::Conversation;
use crate::router::{Decision, RouterTiming};
use crate::scoring::{Estimate, Weights};
use crate::usage::Usage;

/// One routing decision: the prediction for every candidate and, once known, the
/// measured outcome of the chosen one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct DecisionRecord {
    pub decision_id: String,
    /// When the decision was made, in milliseconds since the Unix epoch.
    pub created_at_ms: u64,
    pub api: ApiFormat,
    /// `probe`, or `affinity` for provider-stored state.
    pub router: String,
    /// The chosen model.
    pub model: String,
    /// The provider's name for the chosen model.
    pub upstream_model: Option<String>,
    pub reason: String,
    pub weights: Option<Weights>,
    /// The prediction for the chosen model.
    pub predicted: Option<Estimate>,
    /// Every candidate's prediction and scores, best first.
    pub candidates: Vec<Estimate>,
    pub conversation: Option<Conversation>,
    /// How long the decision took.
    pub timing: Option<RouterTiming>,
    /// What happened, once the response has finished (or the client disconnected);
    /// `null` while it is still streaming.
    pub actual: Option<Actual>,
}

/// The measured outcome of a forwarded request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Actual {
    /// The provider's HTTP status.
    pub status: u16,
    /// Whether the whole response was delivered (false if the client disconnected or
    /// the upstream stream failed).
    pub complete: bool,
    /// Token usage as the provider reported it, if it did (OpenAI Chat streams report
    /// it only with `stream_options.include_usage`).
    pub usage: Option<Usage>,
    /// Cost of `usage` at the model's prices, in USD.
    pub cost_usd: Option<f64>,
    /// From sending the request upstream to the first byte of the response body. For
    /// streamed responses, close to time to first token.
    pub time_to_first_byte_s: f64,
    /// From sending the request upstream to the end of the response.
    pub total_time_s: f64,
    /// Output tokens per second while streaming (after the first byte); only for
    /// streamed responses with usage.
    pub output_tokens_per_sec: Option<f64>,
}

impl DecisionRecord {
    /// A record for `decision` in `api`, with no outcome yet.
    pub fn new(decision: &Decision, api: ApiFormat, upstream_model: Option<String>) -> Self {
        Self {
            decision_id: decision.decision_id.clone(),
            created_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as u64),
            api,
            router: decision.router.clone(),
            model: decision.model.clone(),
            upstream_model,
            reason: decision.reason.clone(),
            weights: decision.weights,
            predicted: decision
                .estimates
                .iter()
                .find(|e| e.model == decision.model)
                .cloned(),
            candidates: decision.estimates.clone(),
            conversation: decision.conversation.clone(),
            timing: decision.timing,
            actual: None,
        }
    }
}

/// A bounded, in-memory log of recent decisions.
pub struct DecisionLog {
    capacity: usize,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    records: HashMap<String, DecisionRecord>,
    order: VecDeque<String>,
}

impl Default for DecisionLog {
    fn default() -> Self {
        Self::new(Self::DEFAULT_CAPACITY)
    }
}

impl DecisionLog {
    pub const DEFAULT_CAPACITY: usize = 10_000;

    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            inner: Mutex::new(Inner::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Add a record, evicting the oldest beyond the capacity.
    pub fn insert(&self, record: DecisionRecord) {
        let mut inner = self.lock();
        let id = record.decision_id.clone();
        if inner.records.insert(id.clone(), record).is_none() {
            inner.order.push_back(id);
        }
        while inner.order.len() > self.capacity {
            if let Some(old) = inner.order.pop_front() {
                inner.records.remove(&old);
            }
        }
    }

    /// Attach the measured outcome to a decision (a no-op if it was evicted).
    pub fn complete(&self, decision_id: &str, actual: Actual) {
        if let Some(r) = self.lock().records.get_mut(decision_id) {
            r.actual = Some(actual);
        }
    }

    pub fn get(&self, decision_id: &str) -> Option<DecisionRecord> {
        self.lock().records.get(decision_id).cloned()
    }

    pub fn len(&self) -> usize {
        self.lock().records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decision(id: &str) -> Decision {
        Decision {
            decision_id: id.into(),
            router: "probe".into(),
            model: "m".into(),
            reason: String::new(),
            conversation: None,
            estimates: vec![],
            weights: None,
            timing: None,
        }
    }

    #[test]
    fn keeps_the_most_recent_and_attaches_outcomes() {
        let log = DecisionLog::new(2);
        for id in ["a", "b", "c"] {
            log.insert(DecisionRecord::new(
                &decision(id),
                ApiFormat::OpenaiChat,
                None,
            ));
        }
        assert_eq!(log.len(), 2);
        assert!(log.get("a").is_none(), "oldest evicted");
        log.complete(
            "c",
            Actual {
                status: 200,
                complete: true,
                usage: None,
                cost_usd: None,
                time_to_first_byte_s: 0.5,
                total_time_s: 1.0,
                output_tokens_per_sec: None,
            },
        );
        assert_eq!(log.get("c").unwrap().actual.unwrap().status, 200);
        log.complete("a", log.get("c").unwrap().actual.unwrap()); // evicted: ignored
        assert!(log.get("b").unwrap().actual.is_none());
    }
}
