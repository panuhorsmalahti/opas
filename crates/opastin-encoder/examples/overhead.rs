//! Measure what activation capture costs: prefill time when capturing all layers, one
//! layer, or none (plain prefill).
//!
//!   cargo run --release -p opastin-encoder --example overhead -- models/gguf/Qwen3.5-0.8B-Q8_0.gguf

use std::time::Duration;

use opastin_encoder::{Encoder, EncoderConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/gguf/Qwen3.5-9B-Q8_0.gguf".into());
    let short = "Rename the variable `x` to `count`.".to_string();
    let long = "Refactor the request handler so that retries use exponential backoff. ".repeat(40);
    let runs = 10;

    let n_layer = {
        let enc = Encoder::load(&path, EncoderConfig::default())?;
        enc.n_layer()
    };
    let mid = n_layer / 2 + 1;
    let setups: [(&str, Option<Vec<usize>>); 3] = [
        ("no capture", Some(vec![])),
        ("1 layer", Some(vec![mid])),
        ("all layers", None),
    ];

    println!("model: {path} ({n_layer} layers), median of {runs} runs");
    for (label, layers) in setups {
        let mut enc = Encoder::load(&path, EncoderConfig { layers, ..EncoderConfig::default() })?;
        for (name, prompt) in [("short", &short), ("long", &long)] {
            enc.encode_prompt(prompt)?; // warm-up for this shape
            let mut times: Vec<Duration> = Vec::new();
            let mut n_tokens = 0;
            for _ in 0..runs {
                let a = enc.encode_prompt(prompt)?;
                n_tokens = a.n_tokens;
                times.push(a.elapsed);
            }
            times.sort();
            println!(
                "  {label:>10} | {name:>5} prompt ({n_tokens:>4} tokens): {:>7.1} ms",
                times[runs / 2].as_secs_f64() * 1000.0
            );
        }
    }
    Ok(())
}
