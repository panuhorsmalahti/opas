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
    /// Path to the TOML config. Default: ./opas.toml if it exists, otherwise
    /// $XDG_CONFIG_HOME/opas/opas.toml (~/.config/opas/opas.toml; on Windows
    /// %APPDATA%\opas\opas.toml), where the installer puts it.
    #[arg(short, long, env = "OPAS_CONFIG")]
    config: Option<PathBuf>,
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
    let path = args.config.map_or_else(default_config_path, Ok)?;
    tracing::info!(config = %path.display(), "loading config");
    let mut config = Config::load(&path)?;
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

/// `./opas.toml`, else the user config file. Errors name both places when neither exists.
fn default_config_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let local = PathBuf::from("opas.toml");
    if local.is_file() {
        return Ok(local);
    }
    let user = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        // Windows: %APPDATA%\opas\opas.toml (where install.ps1 puts it).
        .or_else(|| std::env::var_os("APPDATA").map(PathBuf::from))
        .map(|dir| dir.join("opas").join("opas.toml"));
    match user {
        Some(p) if p.is_file() => Ok(p),
        Some(p) => Err(format!(
            "no config found: create ./opas.toml or {} (scripts/install.sh writes one), or pass --config",
            p.display()
        )
        .into()),
        None => Err("no config found: create ./opas.toml or pass --config".into()),
    }
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
                        "[router] {what} `{}` not found. Download the encoder model (scripts/install.sh does; \
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
