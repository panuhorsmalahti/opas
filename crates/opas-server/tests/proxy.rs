//! End-to-end tests: the opas app in front of a mock upstream on a real TCP port.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Json;
use http_body_util::BodyExt;
use opas::gateway::Gateway;
use opas::{Candidate, Config, Decision, Estimate, RouteError, RouteRequest, Router, Tier};
use serde_json::{json, Value};
use tower::ServiceExt;

/// Mock provider: echoes the model and auth headers it received. Streams SSE when asked.
async fn spawn_upstream(calls: Arc<AtomicUsize>) -> String {
    async fn echo(calls: Arc<AtomicUsize>, headers: HeaderMap, body: Bytes) -> Response {
        let n = calls.fetch_add(1, Ordering::SeqCst);
        let body: Value = serde_json::from_slice(&body).unwrap();
        let h = |n: &str| headers.get(n).map(|v| v.to_str().unwrap().to_owned());
        let id = format!("resp_{n}");
        if body["stream"] == json!(true) && body.get("input").is_some() {
            // Open Responses-style stream: the response id arrives in the first event.
            let sse = format!(
                "event: response.created\ndata: {{\"type\":\"response.created\",\"response\":{{\"id\":\"{id}\",\"model\":\"{}\"}}}}\n\nevent: response.completed\ndata: {{\"type\":\"response.completed\"}}\n\n",
                body["model"].as_str().unwrap()
            );
            return ([("content-type", "text/event-stream")], sse).into_response();
        }
        if body["stream"] == json!(true) {
            let sse = format!(
                "data: {{\"model\":\"{}\",\"chunk\":1}}\n\ndata: {{\"chunk\":2}}\n\ndata: [DONE]\n\n",
                body["model"].as_str().unwrap()
            );
            return ([("content-type", "text/event-stream")], sse).into_response();
        }
        Json(json!({
            "id": id,
            "model": body["model"],
            "x_goog_api_key": h("x-goog-api-key"),
            "authorization": h("authorization"),
            "x_api_key": h("x-api-key"),
            "anthropic_version": h("anthropic-version"),
        }))
        .into_response()
    }

    let (c1, c2, c3, c4) = (calls.clone(), calls.clone(), calls.clone(), calls);
    let app = axum::Router::new()
        .route(
            "/v1/chat/completions",
            post(move |h, b| echo(c1.clone(), h, b)),
        )
        .route("/v1/messages", post(move |h, b| echo(c2.clone(), h, b)))
        .route("/v1/responses", post(move |h, b| echo(c3.clone(), h, b)))
        .route("/v1/interactions", post(move |h, b| echo(c4.clone(), h, b)));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}/v1")
}

struct Harness {
    app: axum::Router,
    gateway: Arc<Gateway>,
    upstream_base: String,
    upstream_calls: Arc<AtomicUsize>,
}

/// Test router that predicts like the probe router, from fixed per-tier numbers:
/// small = 45% success, large = 88%; costs and times from tier defaults.
struct FixedPredictions;

#[async_trait::async_trait]
impl Router for FixedPredictions {
    fn name(&self) -> &str {
        "fixed"
    }

    async fn route(&self, req: &RouteRequest, cands: &[Candidate]) -> Result<Decision, RouteError> {
        let mut estimates: Vec<Estimate> = cands
            .iter()
            .filter_map(|c| {
                let tier = c.tier?;
                let (p, out) = if tier >= Tier::Large {
                    (0.88, 800.0)
                } else {
                    (0.45, 300.0)
                };
                let price = c.effective_pricing()?;
                let speed = c.effective_speed()?;
                Some(Estimate {
                    model: c.id.clone(),
                    tier: Some(tier),
                    p_success: p,
                    expected_cost_usd: (1000.0 * price.input + out * price.output) / 1e6,
                    expected_response_time_s: speed.response_time_s(out),
                    expected_output_tokens: out,
                    cached_prompt_tokens: 0,
                    quality_score: 0.0,
                    speed_score: 0.0,
                    cost_score: 0.0,
                    score: 0.0,
                })
            })
            .collect();
        let weights = req.weights.unwrap_or_default();
        opas::scoring::score(&mut estimates, weights);
        let best = estimates.first().ok_or(RouteError::NoCandidates)?;
        Ok(Decision {
            decision_id: "d".into(),
            router: self.name().into(),
            model: best.model.clone(),
            reason: "fixed".into(),
            conversation: None,
            weights: Some(weights),
            estimates,
        })
    }
}

async fn harness() -> Harness {
    harness_with(None).await
}

async fn harness_with(router: Option<Arc<dyn Router>>) -> Harness {
    let calls = Arc::new(AtomicUsize::new(0));
    let base = spawn_upstream(calls.clone()).await;
    let config = Config::from_toml_str(&format!(
        r#"
        router_model = "auto"
        [router]
        kind = "random"
        seed = 1

        [[models]]
        id = "big"
        tier = "large"
        upstream_model = "big-upstream"
        openai = {{ base_url = "{base}" }}
        anthropic = {{ base_url = "{base}" }}
        responses = {{ base_url = "{base}" }}
        interactions = {{ base_url = "{base}" }}

        [[models]]
        id = "small"
        tier = "small"
        openai = {{ base_url = "{base}" }}
        anthropic = {{ base_url = "{base}" }}
        responses = {{ base_url = "{base}" }}
        interactions = {{ base_url = "{base}" }}

        [[models]]
        id = "openai-only"
        openai = {{ base_url = "{base}" }}
        "#
    ))
    .unwrap();
    let gateway = Arc::new(match router {
        Some(r) => Gateway::with_router(config, r),
        None => Gateway::new(config).unwrap(),
    });
    Harness {
        app: opas_server::app(Arc::clone(&gateway)),
        gateway,
        upstream_base: base,
        upstream_calls: calls,
    }
}

async fn send(app: &axum::Router, req: Request<Body>) -> (StatusCode, HeaderMap, Bytes) {
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let headers = res.headers().clone();
    let body = res.into_body().collect().await.unwrap().to_bytes();
    (status, headers, body)
}

fn req(path: &str, body: Value, headers: &[(&str, &str)]) -> Request<Body> {
    let mut b = Request::post(path).header("content-type", "application/json");
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    b.body(Body::from(body.to_string())).unwrap()
}

fn json_body(b: &Bytes) -> Value {
    serde_json::from_slice(b).unwrap()
}

#[tokio::test]
async fn openai_router_model_is_routed_and_proxied() {
    let h = harness().await;
    let (status, headers, body) = send(
        &h.app,
        req(
            "/v1/chat/completions",
            json!({"model": "auto", "messages": [{"role": "user", "content": "hi"}]}),
            &[("authorization", "Bearer client-token")],
        ),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let model = headers["x-opas-model"].to_str().unwrap();
    assert!(["big", "small", "openai-only"].contains(&model), "{model}");
    assert_eq!(headers["x-opas-router"], "random");
    assert!(headers.contains_key("x-opas-decision-id"));

    let body = json_body(&body);
    let expected = if model == "big" {
        "big-upstream"
    } else {
        model
    };
    assert_eq!(
        body["model"], expected,
        "upstream received the upstream model name"
    );
    assert_eq!(body["authorization"], "Bearer client-token");
}

#[tokio::test]
async fn anthropic_routes_only_between_models_with_an_anthropic_upstream() {
    let h = harness().await;
    let mut seen = HashSet::new();
    for _ in 0..100 {
        let (status, headers, body) = send(
            &h.app,
            req(
                "/v1/messages",
                json!({"model": "auto", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}),
                &[("x-api-key", "client-key")],
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let body = json_body(&body);
        assert_eq!(body["x_api_key"], "client-key");
        assert_eq!(
            body["anthropic_version"], "2023-06-01",
            "default version added"
        );
        seen.insert(headers["x-opas-model"].to_str().unwrap().to_owned());
    }
    assert_eq!(seen, HashSet::from(["big".to_owned(), "small".to_owned()]));
}

#[tokio::test]
async fn named_model_goes_direct_without_a_decision() {
    let h = harness().await;
    let (status, headers, body) = send(
        &h.app,
        req(
            "/v1/messages",
            json!({"model": "small", "max_tokens": 10, "messages": []}),
            &[("anthropic-version", "2026-10-21")],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["x-opas-model"], "small");
    assert!(!headers.contains_key("x-opas-decision-id"));
    let body = json_body(&body);
    assert_eq!(body["model"], "small");
    assert_eq!(body["anthropic_version"], "2026-10-21");
}

#[tokio::test]
async fn streaming_responses_pass_through() {
    let h = harness().await;
    let (status, headers, body) = send(
        &h.app,
        req(
            "/v1/chat/completions",
            json!({"model": "big", "stream": true, "messages": [{"role": "user", "content": "hi"}]}),
            &[],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "text/event-stream");
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("\"model\":\"big-upstream\""), "{text}");
    assert!(text.contains("\"chunk\":2"));
    assert!(text.ends_with("data: [DONE]\n\n"));
}

#[tokio::test]
async fn errors_use_each_apis_error_shape() {
    let h = harness().await;

    let (status, _, body) = send(
        &h.app,
        req(
            "/v1/chat/completions",
            json!({"model": "nope", "messages": []}),
            &[],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let body = json_body(&body);
    assert_eq!(body["error"]["type"], "not_found_error");
    assert!(body["error"]["message"].as_str().unwrap().contains("nope"));

    let (status, _, body) = send(
        &h.app,
        req(
            "/v1/messages",
            json!({"model": "openai-only", "messages": []}),
            &[],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body = json_body(&body);
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");

    let bad = Request::post("/v1/messages")
        .body(Body::from("{not json"))
        .unwrap();
    let (status, _, body) = send(&h.app, bad).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json_body(&body)["type"], "error");

    let (status, _, _) = send(
        &h.app,
        req("/v1/chat/completions", json!({"model": "auto"}), &[]),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "missing messages");

    assert_eq!(
        h.upstream_calls.load(Ordering::SeqCst),
        0,
        "no error reached upstream"
    );
}

#[tokio::test]
async fn route_endpoint_returns_decision_without_calling_upstream() {
    let h = harness().await;
    let (status, _, body) = send(
        &h.app,
        req(
            "/v1/route",
            json!({
                "api": "anthropic_messages",
                "request": {"model": "auto", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}
            }),
            &[],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = json_body(&body);
    assert_eq!(body["router"], "random");
    assert_eq!(body["api"], "anthropic_messages");
    let model = body["model"].as_str().unwrap();
    assert!(model == "big" || model == "small", "{model}");
    let expected = if model == "big" {
        "big-upstream"
    } else {
        "small"
    };
    assert_eq!(body["upstream_model"], expected);
    assert!(body["decision_id"].as_str().is_some());
    assert_eq!(h.upstream_calls.load(Ordering::SeqCst), 0);

    let (status, _, _) = send(
        &h.app,
        req("/v1/route", json!({"api": "grpc", "request": {}}), &[]),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn models_endpoint_lists_router_and_configured_models() {
    let h = harness().await;
    let get = |anthropic: bool| {
        let mut b = Request::get("/v1/models");
        if anthropic {
            b = b.header("anthropic-version", "2023-06-01");
        }
        b.body(Body::empty()).unwrap()
    };

    let (status, _, body) = send(&h.app, get(false)).await;
    assert_eq!(status, StatusCode::OK);
    let body = json_body(&body);
    assert_eq!(body["object"], "list");
    let ids: Vec<_> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].clone())
        .collect();
    assert_eq!(
        ids,
        [
            json!("auto"),
            json!("big"),
            json!("small"),
            json!("openai-only")
        ]
    );

    let (_, _, body) = send(&h.app, get(true)).await;
    let body = json_body(&body);
    assert_eq!(body["data"][0]["type"], "model");
    assert_eq!(body["has_more"], false);
}

#[tokio::test]
async fn cache_aware_routing_keeps_a_conversation_on_one_model() {
    let h = harness().await;
    // Ten independent conversations: each turn 2 must go to the model that served turn 1,
    // even though the random choice would otherwise pick at random.
    for i in 0..10 {
        let first_user =
            json!({"role": "user", "content": format!("task {i}: refactor the parser")});
        let (status, headers, _) = send(
            &h.app,
            req(
                "/v1/messages",
                json!({"model": "auto", "max_tokens": 10, "system": "agent", "messages": [first_user.clone()]}),
                &[],
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let first = headers["x-opas-model"].to_str().unwrap().to_owned();
        assert!(
            !headers.contains_key("x-opas-previous-model"),
            "turn 1 is new"
        );

        let (status, headers, _) = send(
            &h.app,
            req(
                "/v1/messages",
                json!({"model": "auto", "max_tokens": 10, "system": "agent", "messages": [
                    first_user,
                    {"role": "assistant", "content": "Looking at parser.rs"},
                    {"role": "user", "content": "go on"}
                ]}),
                &[],
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            headers["x-opas-model"],
            first.as_str(),
            "turn 2 stays on the cached model"
        );
        assert_eq!(headers["x-opas-previous-model"], first.as_str());
    }
}

#[tokio::test]
async fn route_endpoint_reports_the_continued_conversation() {
    let h = harness().await;
    let turn1 = json!({"model": "auto", "messages": [{"role": "user", "content": "hello"}]});
    let (_, _, body) = send(
        &h.app,
        req(
            "/v1/route",
            json!({"api": "openai_chat", "request": turn1}),
            &[],
        ),
    )
    .await;
    let first = json_body(&body)["model"].as_str().unwrap().to_owned();
    assert!(json_body(&body)
        .get("conversation")
        .map_or(true, Value::is_null));

    let turn2 = json!({"model": "auto", "messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "more"}
    ]});
    let (_, _, body) = send(
        &h.app,
        req(
            "/v1/route",
            json!({"api": "openai_chat", "request": turn2}),
            &[],
        ),
    )
    .await;
    let body = json_body(&body);
    assert_eq!(body["model"], first.as_str());
    assert_eq!(body["conversation"]["previous_model"], first.as_str());
    assert!(body["conversation"]["cached_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn open_responses_follow_ups_return_to_the_model_holding_the_state() {
    let h = harness().await;
    for stream in [false, true] {
        for _ in 0..8 {
            let first = json!({"model": "auto", "instructions": "be brief", "input": "start a task", "stream": stream});
            let (status, headers, body) = send(
                &h.app,
                req("/v1/responses", first, &[("authorization", "Bearer k")]),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            let model = headers["x-opas-model"].to_str().unwrap().to_owned();
            let text = String::from_utf8(body.to_vec()).unwrap();
            let id = text
                .split("resp_")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap();
            let id = format!("resp_{id}");

            // Only the new input; the provider holds the rest under previous_response_id.
            let follow = json!({"model": "auto", "previous_response_id": id, "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "continue"}]}
            ]});
            let (status, headers, _) = send(&h.app, req("/v1/responses", follow, &[])).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(headers["x-opas-model"], model.as_str(), "stream={stream}");
            assert_eq!(headers["x-opas-router"], "affinity");
        }
    }
}

#[tokio::test]
async fn gemini_interactions_are_routed_with_google_auth_and_errors() {
    let h = harness().await;
    let body = json!({"model": "auto", "system_instruction": "x", "input": [
        {"type": "user_input", "content": [{"type": "text", "text": "hello"}]}
    ]});
    for path in ["/v1beta/interactions", "/v1/interactions"] {
        let (status, headers, resp) =
            send(&h.app, req(path, body.clone(), &[("x-goog-api-key", "gk")])).await;
        assert_eq!(status, StatusCode::OK, "{path}");
        let model = headers["x-opas-model"].to_str().unwrap();
        assert!(model == "big" || model == "small");
        let resp = json_body(&resp);
        assert_eq!(resp["x_goog_api_key"], "gk", "client key passed through");
        assert!(resp["id"].as_str().unwrap().starts_with("resp_"));
    }

    // Google-style error body for an unknown model.
    let (status, _, resp) = send(
        &h.app,
        req(
            "/v1beta/interactions",
            json!({"model": "nope", "input": "x"}),
            &[],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let resp = json_body(&resp);
    assert_eq!(resp["error"]["status"], "NOT_FOUND");
    assert_eq!(resp["error"]["code"], 404);

    // Open Responses error vocabulary.
    let (status, _, resp) = send(&h.app, req("/v1/responses", json!({"model": "auto"}), &[])).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json_body(&resp)["error"]["type"], "invalid_request");
}

#[tokio::test]
async fn route_endpoint_accepts_consumer_tier_candidates_and_new_apis() {
    let h = harness().await;
    let body = json!({
        "api": "open_responses",
        "request": {"model": "auto", "input": "summarize this"},
        "candidates": [
            {"id": "my-haiku", "tier": "small", "pricing": {"input": 0.1, "cached_input": 0.01, "output": 0.5}},
            {"id": "my-fable", "tier": "xlarge"}
        ]
    });
    let (status, _, resp) = send(&h.app, req("/v1/route", body, &[])).await;
    assert_eq!(status, StatusCode::OK);
    let resp = json_body(&resp);
    let model = resp["model"].as_str().unwrap();
    assert!(model == "my-haiku" || model == "my-fable", "{model}");
    assert!(resp["upstream_model"].is_null(), "not a configured model");

    let gemini = json!({"api": "gemini_interactions", "request": {"model": "auto", "input": "hi"}});
    let (status, _, _) = send(&h.app, req("/v1/route", gemini, &[])).await;
    assert_eq!(status, StatusCode::OK);
    let empty = json!({"api": "openai_chat", "request": {"messages": []}, "candidates": []});
    let (status, _, _) = send(&h.app, req("/v1/route", empty, &[])).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn consumers_set_quality_speed_cost_weights_and_get_estimates() {
    let h = harness_with(Some(Arc::new(FixedPredictions))).await;
    let route = |extra: Value| {
        let mut body = json!({"api": "openai_chat", "request": {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        req("/v1/route", body, &[])
    };

    // Default weights: quality 60 / speed 20 / cost 20, echoed back with estimates.
    let (status, _, body) = send(&h.app, route(json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    let body = json_body(&body);
    assert_eq!(
        body["weights"],
        json!({"quality": 60.0, "speed": 20.0, "cost": 20.0})
    );
    let est = body["estimates"].as_array().unwrap();
    assert_eq!(
        est.len(),
        2,
        "big (large) and small (small); openai-only has no tier"
    );
    for e in est {
        for k in [
            "p_success",
            "expected_cost_usd",
            "expected_response_time_s",
            "quality_score",
            "speed_score",
            "cost_score",
            "score",
        ] {
            assert!(e[k].is_number(), "{k} in {e}");
        }
    }
    assert!(
        est[0]["score"].as_f64() >= est[1]["score"].as_f64(),
        "best first"
    );

    // Presets and explicit weights change the choice.
    let pick = |b: &Bytes| json_body(b)["model"].as_str().unwrap().to_owned();
    let (_, _, b) = send(&h.app, route(json!({"preference": "quality"}))).await;
    assert_eq!(pick(&b), "big");
    let (_, _, b) = send(&h.app, route(json!({"preference": "cost"}))).await;
    assert_eq!(pick(&b), "small");
    let (_, _, b) = send(&h.app, route(json!({"weights": {"speed": 1}}))).await;
    assert_eq!(pick(&b), "small");
    let (_, _, b) = send(&h.app, route(json!({"weights": {"quality": 1}}))).await;
    assert_eq!(pick(&b), "big");
    let (status, _, _) = send(&h.app, route(json!({"weights": {"quality": 0}}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "all-zero weights");
    let (status, _, _) = send(&h.app, route(json!({"preference": "fastest"}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn proxy_requests_take_weights_from_headers_and_report_the_estimate() {
    let h = harness_with(Some(Arc::new(FixedPredictions))).await;
    let body =
        json!({"model": "auto", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]});
    let (status, headers, _) = send(
        &h.app,
        req(
            "/v1/messages",
            body.clone(),
            &[("x-opas-preference", "quality")],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["x-opas-model"], "big");
    let est = headers["x-opas-estimate"].to_str().unwrap();
    assert!(
        est.contains("p_success=0.880")
            && est.contains("cost_usd=")
            && est.contains("response_time_s="),
        "{est}"
    );

    let (_, headers, _) = send(
        &h.app,
        req(
            "/v1/messages",
            body.clone(),
            &[("x-opas-weights", "cost=1")],
        ),
    )
    .await;
    assert_eq!(headers["x-opas-model"], "small");

    let (status, _, resp) = send(
        &h.app,
        req("/v1/messages", body, &[("x-opas-weights", "quality=abc")]),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json_body(&resp)["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn consumers_override_prices_per_request() {
    let h = harness_with(Some(Arc::new(FixedPredictions))).await;
    let msg =
        json!({"model": "auto", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]});
    let route = |pricing: Value| {
        req(
            "/v1/route",
            json!({"api": "anthropic_messages", "request": msg.clone(), "preference": "cost", "pricing": pricing}),
            &[],
        )
    };
    // Cost preference picks the cheap small model at its tier-default price...
    let (_, _, b) = send(&h.app, route(json!({}))).await;
    assert_eq!(json_body(&b)["model"], "small");
    // ...but not when this request says small is expensive.
    let (status, _, b) = send(
        &h.app,
        route(json!({"small": {"input": 500, "cached_input": 500, "output": 900}})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let b = json_body(&b);
    assert_eq!(b["model"], "big");
    let small = b["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["model"] == "small")
        .unwrap()
        .clone();
    assert!(
        small["expected_cost_usd"].as_f64().unwrap() > 0.5,
        "{small}"
    );

    // The same through a proxy header.
    let (_, headers, _) = send(
        &h.app,
        req(
            "/v1/messages",
            msg.clone(),
            &[
                ("x-opas-preference", "cost"),
                ("x-opas-pricing", "small=500/500/900"),
            ],
        ),
    )
    .await;
    assert_eq!(headers["x-opas-model"], "big");
    let (status, _, _) = send(
        &h.app,
        req("/v1/messages", msg, &[("x-opas-pricing", "small=cheap")]),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn prices_come_from_models_dev_for_operators_and_consumers() {
    let h = harness_with(Some(Arc::new(FixedPredictions))).await;
    let table = |b: &Bytes| json_body(b)["models"].as_array().unwrap().clone();
    let source =
        |rows: &[Value], m: &str| rows.iter().find(|r| r["model"] == m).unwrap()["source"].clone();

    // Before a catalog is loaded: tier defaults, or unknown for a model without a tier.
    let (status, _, b) = send(
        &h.app,
        Request::get("/v1/pricing").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let rows = table(&b);
    assert_eq!(source(&rows, "big"), "tier_default");
    assert_eq!(source(&rows, "openai-only"), "unknown");

    // A catalog whose provider API host matches big's upstream: big's upstream model
    // "big-upstream" is found by inference.
    h.gateway
        .set_catalog(opas::PriceCatalog::from_models_dev(&json!({
            "mockprov": {"api": h.upstream_base.clone(), "models": {
                "big-upstream": {"cost": {"input": 3, "output": 15, "cache_read": 0.3}}
            }}
        })));
    let (_, _, b) = send(
        &h.app,
        Request::get("/v1/pricing").body(Body::empty()).unwrap(),
    )
    .await;
    let rows = table(&b);
    let big = rows.iter().find(|r| r["model"] == "big").unwrap();
    assert_eq!(big["source"], "models_dev");
    assert_eq!(big["models_dev"], "mockprov/big-upstream");
    assert_eq!(big["pricing"]["output"], 15.0);

    // Consumers can price their own candidates by models.dev key.
    let body = json!({
        "api": "openai_chat",
        "request": {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
        "preference": "cost",
        "candidates": [
            {"id": "x", "tier": "small", "models_dev": "mockprov/big-upstream"},
            {"id": "y", "tier": "small"}
        ]
    });
    let (_, _, b) = send(&h.app, req("/v1/route", body, &[])).await;
    let b = json_body(&b);
    assert_eq!(
        b["model"], "y",
        "x is priced at 3/15 from models.dev, y at the small tier default"
    );
    let x = b["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["model"] == "x")
        .unwrap()
        .clone();
    let y = b["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["model"] == "y")
        .unwrap()
        .clone();
    assert!(x["expected_cost_usd"].as_f64() > y["expected_cost_usd"].as_f64());
}
