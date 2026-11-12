use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use opas::gateway::Gateway;
use opas::{Config, RouterConfig};
use tracing_subscriber::EnvFilter;

/// opas: an LLM router with OpenAI- and Anthropic-compatible endpoints.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Path to the TOML config.
    #[arg(short, long, env = "OPAS_CONFIG", default_value = "opas.toml")]
    config: PathBuf,
    /// Listen address; overrides `listen` in the config.
    #[arg(short, long, env = "OPAS_LISTEN")]
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
        let mut vars: Vec<&str> = [
            &model.openai,
            &model.anthropic,
            &model.responses,
            &model.interactions,
        ]
        .into_iter()
        .flatten()
        .filter_map(|u| u.api_key_env.as_deref())
        .filter(|var| std::env::var_os(var).is_none())
        .collect();
        vars.sort_unstable();
        vars.dedup();
        for var in vars {
            tracing::warn!(model = %model.id, %var, "API key variable is not set; requests to this model will fail");
        }
    }

    let listen = config.listen.clone();
    let gateway = build_gateway(config)?;
    tracing::info!(
        router = gateway.router().map_or("none (plain proxy)", |r| r.name()),
        router_model = %gateway.config().router_model,
        cache_aware = gateway.config().cache.enabled,
        models = ?gateway.config().models.iter().map(|m| &m.id).collect::<Vec<_>>(),
        "starting"
    );

    let gateway = Arc::new(gateway);
    if gateway.config().pricing.models_dev {
        if let Some(n) = gateway.load_price_cache() {
            tracing::info!(models = n, "loaded cached models.dev prices");
        }
        let gw = Arc::clone(&gateway);
        let every = std::time::Duration::from_secs(gw.config().pricing.refresh_hours.max(1) * 3600);
        tokio::spawn(async move {
            loop {
                match gw.refresh_prices().await {
                    Ok(n) => tracing::info!(models = n, "loaded models.dev prices"),
                    Err(e) => {
                        tracing::warn!(error = %e, "could not load models.dev prices; using config and tier defaults")
                    }
                }
                tokio::time::sleep(every).await;
            }
        });
    }

    let listener = tokio::net::TcpListener::bind(&listen).await?;
    let addr = listener.local_addr()?;
    tracing::info!("listening on http://{addr}");
    if !addr.ip().is_loopback() {
        tracing::warn!(
            %addr,
            "listening on a non-loopback address: opas has no client authentication, so anyone who can reach \
             this port can use the configured provider keys. Keep it on a private network (in Docker: \
             -p 127.0.0.1:8080:8080) or put an authenticating proxy in front"
        );
    }
    axum::serve(listener, opas_server::app(gateway))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

/// The probe router needs the llama.cpp encoder (feature `probe`). Without a `[router]`,
/// opas runs as a plain proxy.
fn build_gateway(config: Config) -> Result<Gateway, Box<dyn std::error::Error>> {
    match &config.router {
        #[cfg(feature = "probe")]
        Some(RouterConfig::Probe {
            encoder_model,
            probe,
            weights,
            n_gpu_layers,
        }) => {
            let untiered: Vec<&str> = config
                .models
                .iter()
                .filter(|m| m.tier.is_none())
                .map(|m| m.id.as_str())
                .collect();
            if !untiered.is_empty() {
                tracing::warn!(models = ?untiered, "models without a `tier` are never chosen by the probe router");
            }
            for (what, path) in [("encoder model", encoder_model), ("probe", probe)] {
                if !path.is_file() {
                    return Err(format!(
                        "[router] {what} `{}` not found. Download the encoder model (see the README quick start; \
                         in Docker, mount it at /models), or remove [router] to run a plain proxy",
                        path.display()
                    )
                    .into());
                }
            }
            tracing::info!(encoder = %encoder_model.display(), probe = %probe.display(), %weights, "loading probe router");
            let router =
                opas_encoder::ProbeRouter::load(encoder_model, probe, *weights, *n_gpu_layers)?;
            Ok(Gateway::with_router(config, Arc::new(router)))
        }
        #[cfg(not(feature = "probe"))]
        Some(RouterConfig::Probe { .. }) => {
            Err("the probe router needs opas-server built with the `probe` feature".into())
        }
        None => {
            tracing::warn!(
                "no [router] configured: running as a plain proxy (requests must name a model)"
            );
            Ok(Gateway::new(config)?)
        }
    }
}
