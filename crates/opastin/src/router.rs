//! The [`Router`] trait and the built-in routers.

use std::sync::Mutex;

use async_trait::async_trait;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::request::RouteRequest;

/// A model the router may choose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    /// Model id as exposed by the router (matches a configured model id).
    pub id: String,
}

impl Candidate {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

impl From<&str> for Candidate {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

/// The outcome of one routing decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    /// Unique id for this decision, for logging and later feedback.
    pub decision_id: String,
    /// Name of the router that decided.
    pub router: String,
    /// The chosen candidate id.
    pub model: String,
    /// Human-readable explanation.
    pub reason: String,
}

#[derive(Debug, thiserror::Error)]
pub enum RouteError {
    #[error("no candidate models to route between")]
    NoCandidates,
    #[error("router failed: {0}")]
    Failed(String),
}

/// Picks one candidate model for a request.
///
/// Async so that future routers can call out to classifiers or other models.
#[async_trait]
pub trait Router: Send + Sync {
    /// Short name, reported in [`Decision::router`].
    fn name(&self) -> &str;

    /// Choose one of `candidates` for `request`.
    async fn route(
        &self,
        request: &RouteRequest,
        candidates: &[Candidate],
    ) -> Result<Decision, RouteError>;
}

/// Picks a candidate uniformly at random. With two candidates, a random choice.
///
/// Useful as a baseline: a real router should beat random routing at equal cost.
pub struct RandomRouter {
    rng: Mutex<StdRng>,
}

impl RandomRouter {
    /// Seeded from OS entropy.
    pub fn new() -> Self {
        Self {
            rng: Mutex::new(StdRng::from_entropy()),
        }
    }

    /// Deterministic sequence of choices, for tests and reproducible benchmarks.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl Default for RandomRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Router for RandomRouter {
    fn name(&self) -> &str {
        "random"
    }

    async fn route(
        &self,
        _request: &RouteRequest,
        candidates: &[Candidate],
    ) -> Result<Decision, RouteError> {
        if candidates.is_empty() {
            return Err(RouteError::NoCandidates);
        }
        let index = {
            let mut rng = self.rng.lock().unwrap_or_else(|p| p.into_inner());
            rng.gen_range(0..candidates.len())
        };
        Ok(Decision {
            decision_id: uuid::Uuid::new_v4().to_string(),
            router: self.name().to_owned(),
            model: candidates[index].id.clone(),
            reason: format!("uniform random pick from {} candidates", candidates.len()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ApiFormat;

    fn request() -> RouteRequest {
        RouteRequest {
            format: ApiFormat::OpenaiChat,
            model: None,
            system: None,
            messages: vec![],
            has_tools: false,
            stream: false,
        }
    }

    #[tokio::test]
    async fn picks_every_candidate_roughly_evenly() {
        let router = RandomRouter::with_seed(7);
        let candidates = [Candidate::from("a"), Candidate::from("b")];
        let mut a = 0;
        for _ in 0..1000 {
            let d = router.route(&request(), &candidates).await.unwrap();
            assert!(d.model == "a" || d.model == "b");
            if d.model == "a" {
                a += 1;
            }
        }
        assert!((400..=600).contains(&a), "got {a} of 1000 for a");
    }

    #[tokio::test]
    async fn same_seed_same_choices() {
        let candidates = [
            Candidate::from("a"),
            Candidate::from("b"),
            Candidate::from("c"),
        ];
        let (r1, r2) = (RandomRouter::with_seed(42), RandomRouter::with_seed(42));
        for _ in 0..50 {
            let d1 = r1.route(&request(), &candidates).await.unwrap();
            let d2 = r2.route(&request(), &candidates).await.unwrap();
            assert_eq!(d1.model, d2.model);
            assert_ne!(d1.decision_id, d2.decision_id);
        }
    }

    #[tokio::test]
    async fn errors_without_candidates() {
        let err = RandomRouter::new()
            .route(&request(), &[])
            .await
            .unwrap_err();
        assert!(matches!(err, RouteError::NoCandidates));
    }
}
