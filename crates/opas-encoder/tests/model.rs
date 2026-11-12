//! Tests against a real GGUF model. They need a model file, so they are skipped unless
//! OPAS_TEST_MODEL points to one, e.g.
//!
//!   OPAS_TEST_MODEL=models/gguf/Qwen3.5-0.8B-Q8_0.gguf cargo test -p opas-encoder --release

use opas::{ApiFormat, RouteRequest};
use opas_encoder::{cosine, Encoder, EncoderConfig, ThinkingMode};
use serde_json::json;

fn model_path() -> Option<String> {
    let path = std::env::var("OPAS_TEST_MODEL").ok()?;
    // Tests run from the crate directory; accept paths relative to the workspace root too.
    let p = std::path::Path::new(&path);
    if p.exists() {
        return Some(path);
    }
    let from_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(p);
    from_root.exists().then(|| from_root.display().to_string())
}

macro_rules! require_model {
    () => {
        match model_path() {
            Some(p) => p,
            None => {
                eprintln!("skipping: set OPAS_TEST_MODEL to a GGUF file");
                return;
            }
        }
    };
}

#[test]
fn captures_every_layer_for_one_prefill() {
    let path = require_model!();
    let mut enc = Encoder::load(&path, EncoderConfig::default()).unwrap();
    let acts = enc
        .encode_prompt("Explain the borrow checker in one sentence.")
        .unwrap();

    assert_eq!(acts.layers, (0..=enc.n_layer()).collect::<Vec<_>>());
    assert_eq!(acts.last.len(), enc.n_layer() + 1);
    for (last, mean) in acts.last.iter().zip(&acts.mean) {
        assert_eq!(last.len(), enc.n_embd());
        assert_eq!(mean.len(), enc.n_embd());
        assert!(last.iter().chain(mean).all(|x| x.is_finite()));
    }
    // Layers transform the residual stream: the last layer differs from the embeddings.
    assert!(cosine(&acts.last[0], &acts.last[enc.n_layer()]) < 0.99);
}

#[test]
fn layer_subset_matches_full_capture_and_is_deterministic() {
    let path = require_model!();
    let prompt = "Design a cache invalidation strategy for a CDN.";
    let mut full = Encoder::load(&path, EncoderConfig::default()).unwrap();
    let a = full.encode_prompt(prompt).unwrap();
    let b = full.encode_prompt(prompt).unwrap();
    for (x, y) in a.last.iter().zip(&b.last) {
        assert!(cosine(x, y) > 0.9999, "repeat run differs");
    }
    drop(full);

    let pick = vec![3, 7];
    let config = EncoderConfig {
        layers: Some(pick.clone()),
        ..EncoderConfig::default()
    };
    let mut subset = Encoder::load(&path, config).unwrap();
    let s = subset.encode_prompt(prompt).unwrap();
    assert_eq!(s.layers, pick);
    for (k, &i) in pick.iter().enumerate() {
        assert!(cosine(&s.last[k], &a.last[i]) > 0.9999);
        assert!(cosine(&s.mean[k], &a.mean[i]) > 0.9999);
    }
}

#[test]
fn long_prompts_span_several_ubatches_and_truncate() {
    let path = require_model!();
    // ~1500 tokens: more than one 512-token ubatch, so accumulation across ubatches runs.
    let long = "The quick brown fox jumps over the lazy dog. ".repeat(150);
    let config = EncoderConfig {
        max_tokens: 1024,
        ..EncoderConfig::default()
    };
    let mut enc = Encoder::load(&path, config).unwrap();
    let acts = enc.encode_prompt(&long).unwrap();
    assert_eq!(acts.n_tokens, 1024, "truncated to max_tokens");
    assert!(acts.mean.iter().flatten().all(|x| x.is_finite()));
}

#[test]
fn request_formats_and_thinking_modes_change_the_features() {
    let path = require_model!();
    let body = json!({"system": "You are a code reviewer.", "max_tokens": 10,
                      "messages": [{"role": "user", "content": "Review this diff."}]});
    let request = RouteRequest::from_json(ApiFormat::AnthropicMessages, &body).unwrap();

    let mut thinking = Encoder::load(&path, EncoderConfig::default()).unwrap();
    let t = thinking.encode_request(&request).unwrap();
    drop(thinking);
    let config = EncoderConfig {
        thinking: ThinkingMode::NonThinking,
        ..EncoderConfig::default()
    };
    let mut plain = Encoder::load(&path, config).unwrap();
    let p = plain.encode_request(&request).unwrap();

    assert!(p.n_tokens > t.n_tokens, "closed think block adds tokens");
    let mid = t.layers.len() / 2;
    assert!(
        cosine(&t.last[mid], &p.last[mid]) < 0.999,
        "final position differs"
    );
}
