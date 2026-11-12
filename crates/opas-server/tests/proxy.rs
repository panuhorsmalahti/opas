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
use opas::Config;
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
    upstream_calls: Arc<AtomicUsize>,
}

async fn harness() -> Harness {
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
    Harness {
        app: opas_server::app(Arc::new(Gateway::new(config).unwrap())),
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
