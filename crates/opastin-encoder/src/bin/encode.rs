//! opastin-encode: extract prefill activations with the in-process llama.cpp encoder.
//!
//! Input is JSONL, one object per line:
//!   {"id": "...", "prompt": "plain user text"}
//!   {"id": "...", "api": "openai_chat" | "anthropic_messages", "request": {...}}
//!
//! Output directory (readable with numpy.load):
//!   last.npy      f32 [n_requests, n_captured_layers, n_embd]  final prompt token
//!   mean.npy      f32 [n_requests, n_captured_layers, n_embd]  mean over prompt tokens
//!   n_tokens.npy  i64 [n_requests]
//!   seconds.npy   f64 [n_requests]                             prefill + capture time
//!   meta.json     ids, captured layer indices, model path, thinking mode

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use opastin::{ApiFormat, RouteRequest};
use opastin_encoder::{cosine, npy, Encoder, EncoderConfig, ThinkingMode};
use serde::Deserialize;
use serde_json::json;

#[derive(Parser)]
#[command(
    version,
    about = "Extract per-layer prefill activations from a GGUF model (no generation)"
)]
struct Args {
    /// GGUF model file.
    #[arg(long, default_value = "models/gguf/Qwen3.5-9B-Q8_0.gguf")]
    model: PathBuf,
    /// JSONL file of requests.
    #[arg(long, conflicts_with_all = ["demo", "prompt"])]
    input: Option<PathBuf>,
    /// Run the built-in demo prompts and print a similarity matrix.
    #[arg(long)]
    demo: bool,
    /// Encode one prompt.
    #[arg(long)]
    prompt: Option<String>,
    /// Directory to write .npy outputs to.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Truncate longer prompts to N tokens (first and last N/2 are kept).
    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,
    /// Comma-separated hidden-state indices to capture (0 = embeddings). Default: all.
    #[arg(long, value_delimiter = ',')]
    layers: Option<Vec<usize>>,
    /// End the prompt with a closed (empty) think block instead of an open one.
    #[arg(long)]
    non_thinking: bool,
    /// Layers to offload to the GPU (negative = all, 0 = CPU only).
    #[arg(long, default_value_t = -1, allow_negative_numbers = true)]
    n_gpu_layers: i32,
    /// Print the formatted prompt and its token count for each item.
    #[arg(long)]
    print_prompt: bool,
    /// Show llama.cpp logs.
    #[arg(long)]
    verbose: bool,
}

#[derive(Deserialize)]
struct Item {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    api: Option<ApiFormat>,
    #[serde(default)]
    request: Option<serde_json::Value>,
}

const DEMO: [(&str, &str); 5] = [
    ("trivial", "What is 2 + 2?"),
    ("lookup", "What is the capital of Australia?"),
    (
        "rename",
        "Rename the variable `x` to `count` in: `let x = 0; x += 1;`",
    ),
    (
        "bugfix",
        "Our Rust async HTTP proxy sometimes truncates SSE streams under load when the client \
         disconnects mid-stream. Find the likely cause in the hyper/axum body forwarding path and \
         propose a fix with tests.",
    ),
    (
        "design",
        "Design a distributed rate limiter for a multi-region API gateway with per-tenant quotas, \
         burst handling, and graceful degradation during partitions. Compare token bucket vs. \
         sliding window, and give the data model and failure modes.",
    ),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let items: Vec<(String, RouteRequest)> = if args.demo {
        DEMO.iter()
            .map(|(id, p)| (id.to_string(), prompt_request(p)))
            .collect()
    } else if let Some(p) = &args.prompt {
        vec![("prompt".into(), prompt_request(p))]
    } else if let Some(path) = &args.input {
        read_items(path)?
    } else {
        return Err("pass --input, --prompt or --demo".into());
    };

    let thinking = if args.non_thinking {
        ThinkingMode::NonThinking
    } else {
        ThinkingMode::Thinking
    };
    let config = EncoderConfig {
        n_gpu_layers: args.n_gpu_layers,
        max_tokens: args.max_tokens,
        layers: args.layers.clone(),
        thinking,
        verbose: args.verbose,
    };

    let t0 = Instant::now();
    let mut enc = Encoder::load(&args.model, config)?;
    eprintln!(
        "loaded {} in {:.1}s: {} layers, hidden size {}",
        args.model.display(),
        t0.elapsed().as_secs_f64(),
        enc.n_layer(),
        enc.n_embd()
    );

    // Warm-up so the first timing excludes one-time GPU pipeline setup.
    enc.encode_prompt("warm up")?;

    let mut results = Vec::with_capacity(items.len());
    for (i, (id, request)) in items.iter().enumerate() {
        if args.print_prompt {
            let text = opastin_encoder::format_request(request, thinking);
            eprintln!(
                "--- {id} ({} tokens)\n{text}\n---",
                enc.tokenize(&text)?.len()
            );
        }
        let acts = enc.encode_request(request)?;
        eprintln!(
            "[{}/{}] {id}: {} tokens, prefill+capture {:.0} ms",
            i + 1,
            items.len(),
            acts.n_tokens,
            acts.elapsed.as_secs_f64() * 1000.0
        );
        results.push(acts);
    }

    if args.demo {
        print_similarity(&items, &results);
    }

    if let Some(out) = &args.out {
        std::fs::create_dir_all(out)?;
        let (n, l, h) = (results.len(), results[0].layers.len(), enc.n_embd());
        let flat = |f: fn(&opastin_encoder::Activations) -> &Vec<Vec<f32>>| -> Vec<f32> {
            results
                .iter()
                .flat_map(|a| f(a).iter().flatten().copied())
                .collect()
        };
        npy::write(out.join("last.npy"), &[n, l, h], &flat(|a| &a.last))?;
        npy::write(out.join("mean.npy"), &[n, l, h], &flat(|a| &a.mean))?;
        let n_tokens: Vec<i64> = results.iter().map(|a| a.n_tokens as i64).collect();
        npy::write(out.join("n_tokens.npy"), &[n], &n_tokens)?;
        let secs: Vec<f64> = results.iter().map(|a| a.elapsed.as_secs_f64()).collect();
        npy::write(out.join("seconds.npy"), &[n], &secs)?;
        let meta = json!({
            "ids": items.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            "layers": results[0].layers,
            "model": args.model,
            "thinking": !args.non_thinking,
            "max_tokens": args.max_tokens,
        });
        std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
        eprintln!("wrote {} ({n} x {l} x {h})", out.display());
    }
    Ok(())
}

fn prompt_request(prompt: &str) -> RouteRequest {
    RouteRequest::from_json(
        ApiFormat::OpenaiChat,
        &json!({"messages": [{"role": "user", "content": prompt}]}),
    )
    .expect("valid request")
}

fn read_items(path: &PathBuf) -> Result<Vec<(String, RouteRequest)>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut items = Vec::new();
    for (i, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let item: Item = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        let id = item.id.unwrap_or_else(|| i.to_string());
        let request = match (item.prompt, item.api, item.request) {
            (Some(p), _, _) => prompt_request(&p),
            (None, Some(api), Some(body)) => {
                RouteRequest::from_json(api, &body).map_err(|e| format!("line {}: {e}", i + 1))?
            }
            _ => return Err(format!("line {}: needs `prompt` or `api` + `request`", i + 1).into()),
        };
        items.push((id, request));
    }
    Ok(items)
}

/// Cosine similarity of last-token states at the middle captured layer: a sanity check
/// that vectors differ by prompt, not a complexity measure.
fn print_similarity(items: &[(String, RouteRequest)], results: &[opastin_encoder::Activations]) {
    let k = results[0].layers.len() / 2;
    println!(
        "\ncosine similarity of last-token states at hidden-state index {}:",
        results[0].layers[k]
    );
    print!("{:>9}", "");
    for (id, _) in items {
        print!(" {:>8}", &id[..id.len().min(8)]);
    }
    println!();
    for (i, (id, _)) in items.iter().enumerate() {
        print!("{:>9}", &id[..id.len().min(8)]);
        for j in 0..items.len() {
            print!(" {:8.3}", cosine(&results[i].last[k], &results[j].last[k]));
        }
        println!();
    }
}
