//! Use opas as a library: score candidate models yourself and let opas rank them by the
//! quality / speed / cost weights, then call the chosen model yourself.
//!
//!     cargo run -p opas --example decide
//!
//! The router that predicts these numbers from a model's activations is
//! `opas_encoder::ProbeRouter`; here the predictions are typed in by hand.

use opas::scoring::score;
use opas::{Estimate, Preference, Weights};

fn estimate(model: &str, p_success: f64, cost: f64, seconds: f64) -> Estimate {
    Estimate {
        model: model.into(),
        tier: None,
        p_success,
        expected_cost_usd: cost,
        expected_response_time_s: seconds,
        expected_output_tokens: 0.0,
        cached_prompt_tokens: 0,
        quality_score: 0.0,
        speed_score: 0.0,
        cost_score: 0.0,
        score: 0.0,
    }
}

fn main() {
    let options = || {
        vec![
            estimate("claude-haiku-5-5", 0.45, 0.0005, 2.0),
            estimate("claude-sonnet-5-5", 0.78, 0.006, 6.0),
            estimate("claude-opus-5-5", 0.86, 0.02, 11.0),
        ]
    };
    let presets = [
        (
            "default (quality 60 / speed 20 / cost 20)",
            Weights::default(),
        ),
        ("quality", Preference::Quality.weights()),
        ("cost", Preference::Cost.weights()),
        ("speed", Preference::Speed.weights()),
    ];
    for (name, weights) in presets {
        let mut e = options();
        score(&mut e, weights);
        println!("{name:<44} -> {} (score {:.3})", e[0].model, e[0].score);
    }
}
