# Using opas with OpenAI Codex

Codex can send its requests through opas, which then picks a model for every turn, for
example a fast GPT model for small steps and a larger one for hard ones.

## What you need

- **opas routing between OpenAI models.** Codex's prompts and tools (the free-form
  `apply_patch` tool, encrypted reasoning) are built for OpenAI's GPT models, and opas
  forwards Codex's Responses API requests as they are, so every candidate must be an
  OpenAI model with a `responses` upstream:

  ```toml
  [[models]]
  id = "gpt-6-luna"
  tier = "small"
  responses = { base_url = "https://api.openai.com/v1", api_key_env = "OPENAI_API_KEY" }

  [[models]]
  id = "gpt-6.1-sol"
  tier = "medium"
  responses = { base_url = "https://api.openai.com/v1", api_key_env = "OPENAI_API_KEY" }

  [[models]]
  id = "gpt-6-astra"
  tier = "large"
  responses = { base_url = "https://api.openai.com/v1", api_key_env = "OPENAI_API_KEY" }
  ```

- **An OpenAI API key** where opas runs (`OPENAI_API_KEY`). Codex's ChatGPT sign-in only
  works against OpenAI's own endpoint, not against a custom provider.

## Set it up with one command

With opas running:

```sh
curl -fsSL https://raw.githubusercontent.com/panuhorsmalahti/opas/main/scripts/codex.sh | bash
codex --profile opas
```

This writes a Codex profile, `~/.codex/opas.config.toml`, and leaves `config.toml` alone:
plain `codex` keeps working as before, `codex --profile opas` goes through opas.

| Option | Meaning |
|---|---|
| `--default` | route every Codex session through opas: the settings go into `~/.codex/config.toml` (between `# >>> opas` markers) |
| `--preference NAME` | send `x-opas-preference: NAME` (`quality`, `balanced`, `cost`, `speed`) |
| `--url URL` | opas address (default `http://127.0.0.1:8080`) |
| `--auth own` | send your own `OPENAI_API_KEY` through opas instead of a placeholder (the opas config must then not set `api_key_env`) |
| `--codex-home DIR` | another Codex directory (default `$CODEX_HOME` or `~/.codex`) |
| `--dry-run` / `--uninstall` | print the result / remove what the script added |

## Or configure it yourself

`~/.codex/opas.config.toml` (used with `codex --profile opas`), or the same lines at the
top of `~/.codex/config.toml` to make it the default:

```toml
model_provider = "opas"
model = "auto"
model_catalog_json = "/Users/you/.codex/opas-catalog.json"

[model_providers.opas]
name = "opas"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
experimental_bearer_token = "opas-local"   # placeholder: opas uses its own OpenAI key
capabilities = { remote_compaction = "unsupported" }
# http_headers = { "x-opas-preference" = "cost" }
```

- `wire_api = "responses"` is the only API Codex speaks to custom providers.
- **The model catalog.** Codex enables tools and settings per model name, and would
  treat `auto` as an unknown model (no `apply_patch` tool, generic prompt). The catalog
  [`scripts/codex/opas-catalog.json`](../../scripts/codex/opas-catalog.json) declares
  `auto` with the settings of `gpt-6.1-sol`, copied from Codex's own catalog
  (openai/codex at commit `5cb38e7`, Apache-2.0). Download it next to your config.
- `remote_compaction = "unsupported"` makes Codex compact long sessions itself instead
  of asking the provider to.

## How opas handles Codex traffic

- **Every request carries the whole conversation** (`store: false`), so opas sees each
  turn and can route it; nothing is stored at the provider.
- **Threads.** Codex sends a `thread-id` header, and opas uses it to recognize the turns
  of one session even when the history changes, so the model that holds the warm prompt
  cache is known (any client can send `x-opas-conversation-id` for the same).
- **Switching models mid-session.** Codex sends earlier turns' reasoning as encrypted
  items that only the model that wrote them can read. When opas moves a session to a
  different model, it drops those items (and logs `dropped reasoning items`), so the new
  model continues from the visible conversation instead of rejecting the request.
- **Routing time.** Codex's system prompt is long, so the encoder reads its full window
  (1,024 tokens) on every turn. Measured: 4.3 s per routed turn on an M3 Pro (Metal),
  more on CPU. Codex turns usually take far longer than that, but it is added to each
  one; see [hardware](hardware.md).

## Limits

- Claude or other non-OpenAI models are not supported as Codex targets: opas does not
  translate between the Responses and Messages APIs.
- Codex shows `auto`, not the chosen model; the opas log and `GET /v1/decisions/{id}`
  have each turn's choice, prediction and actual cost.
- The catalog entry follows Codex's catalog format of November 2026; a much newer Codex
  may need an updated `opas-catalog.json` (rerun the script).
