use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use opastin::gateway::Gateway;
use opastin::{Config, RouterConfig};
use tracing_subscriber::EnvFilter;

/// opastin: an LLM router with OpenAI- and Anthropic-compatible endpoints.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Path to the TOML config.
    #[arg(short, long, env = "OPASTIN_CONFIG", default_value = "opastin.toml")]
    config: PathBuf,
    /// Listen address; overrides `listen` in the config.
    #[arg(short, long, env = "OPASTIN_LISTEN")]
    listen: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let mut config = Config::load(&args.config)?;
    if let Some(listen) = args.listen {
        config.listen = listen;
    }

    for model in &config.models {
        for upstream in [&model.openai, &model.anthropic].into_iter().flatten() {
            if let Some(var) = &upstream.api_key_env {
                if std::env::var_os(var).is_none() {
                    tracing::warn!(model = %model.id, %var, "API key variable is not set; requests to this upstream will fail");
                }
            }
        }
    }

    let listen = config.listen.clone();
    let gateway = build_gateway(config)?;
    tracing::info!(
        router = gateway.router().name(),
        router_model = %gateway.config().router_model,
        cache_aware = gateway.config().cache.enabled,
        models = ?gateway.config().models.iter().map(|m| &m.id).collect::<Vec<_>>(),
        "starting"
    );

    let listener = tokio::net::TcpListener::bind(&listen).await?;
    tracing::info!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, opastin_server::app(Arc::new(gateway)))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

/// The probe router needs the llama.cpp encoder (feature `probe`); other routers are
/// built by the core crate.
fn build_gateway(config: Config) -> Result<Gateway, Box<dyn std::error::Error>> {
    match &config.router {
        #[cfg(feature = "probe")]
        RouterConfig::Probe {
            encoder_model,
            probe,
            lambda,
            n_gpu_layers,
        } => {
            let heads = config
                .models
                .iter()
                .map(|m| (m.id.clone(), m.probe_head.clone().unwrap_or_else(|| m.id.clone())))
                .collect();
            tracing::info!(encoder = %encoder_model.display(), probe = %probe.display(), lambda, "loading probe router");
            let router = opastin_encoder::ProbeRouter::load(encoder_model, probe, heads, *lambda, *n_gpu_layers)?;
            Ok(Gateway::with_router(config, Arc::new(router)))
        }
        #[cfg(not(feature = "probe"))]
        RouterConfig::Probe { .. } => {
            Err("the probe router needs opastin-server built with the `probe` feature".into())
        }
        _ => Ok(Gateway::new(config)?),
    }
}
