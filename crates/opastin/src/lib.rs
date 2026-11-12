//! opastin: model routing for LLM requests.
//!
//! This crate is the SDK. It can be used in two ways:
//!
//! **Decide only.** Ask a [`Router`] which model to use, then make the call yourself:
//!
//! ```
//! use opastin::{ApiFormat, Candidate, RandomRouter, RouteRequest, Router};
//! use serde_json::json;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let body = json!({"model": "auto", "messages": [{"role": "user", "content": "hi"}]});
//! let request = RouteRequest::from_json(ApiFormat::OpenaiChat, &body).unwrap();
//! let candidates = [Candidate::from("claude-opus-5-5"), Candidate::from("claude-haiku-5-5")];
//!
//! let decision = RandomRouter::new().route(&request, &candidates).await.unwrap();
//! println!("use {}", decision.model);
//! # });
//! ```
//!
//! **Route and forward** (feature `gateway`). A [`Gateway`](gateway::Gateway) holds a
//! [`Config`] of models and upstreams, picks a model, and sends the request on. This is
//! what `opastin-server` runs behind its HTTP endpoints.
//!
//! Custom routing logic: implement [`Router`] and pass it to
//! [`Gateway::with_router`](gateway::Gateway::with_router).

pub mod api;
pub mod config;
#[cfg(feature = "gateway")]
pub mod gateway;
pub mod request;
pub mod router;

pub use api::ApiFormat;
pub use config::{Config, ConfigError, ModelConfig, RouterConfig, Upstream};
pub use request::{Message, RequestError, RouteRequest};
pub use router::{Candidate, RandomRouter, Decision, RouteError, Router};
