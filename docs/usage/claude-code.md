# Using opas with Claude Code

Claude Code talks to opas like it talks to Anthropic: opas serves the Anthropic Messages
API, so Claude Code only needs a different base URL and the model name `auto`. opas
then picks a model for every request, such as Haiku for small steps and Opus for hard
ones, and keeps a conversation on the model whose prompt cache is warm unless a switch
is worth it.

## Set it up with one command

Start opas (see the [README](../../README.md#quick-start)), then:

```sh
curl -fsSL https://raw.githubusercontent.com/panuhorsmalahti/opas/main/scripts/claude-code.sh | bash -s -- --default
```

This adds the settings below to `~/.claude/settings.json`, after saving a backup next to
it. `--default` makes `auto` the default model; without it, choose **opas auto** in
`/model` or run `claude --model auto` when you want routing.

| Option | Meaning |
|---|---|
| `--url URL` | opas address (default `http://127.0.0.1:8080`) |
| `--default` | make `auto` the default model |
| `--auth opas` | opas holds the provider key (default; see [Credentials](#credentials)) |
| `--auth own` | keep your own Claude Code login or API key and let opas pass it through |
| `--settings PATH` | another settings file, e.g. a project's `.claude/settings.local.json` |
| `--dry-run` | print the result instead of writing it |
| `--uninstall` | remove what the script added (pass the same `--url` if you changed it) |

The script needs `jq` or `python3`.

## Or configure it yourself

Add this to `~/.claude/settings.json` (merge it with what is already there):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080",
    "ANTHROPIC_AUTH_TOKEN": "opas-local",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "auto",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "opas auto",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": "opas picks the model per request"
  },
  "model": "auto"
}
```

- `ANTHROPIC_BASE_URL` sends Claude Code's API requests to opas. A value in `env` takes
  precedence over the same variable exported in your shell.
- `ANTHROPIC_AUTH_TOKEN` is a placeholder credential so Claude Code does not ask you to
  log in; opas replaces it with the provider key from its own config. Leave it out with
  your own credentials (below).
- `ANTHROPIC_CUSTOM_MODEL_OPTION` adds `auto` to the `/model` picker. Claude Code
  rejects unknown names typed into `/model`, so without this entry only
  `claude --model auto` and the `model` setting reach it.
- `"model": "auto"` makes routing the default. Leave it out to opt in per session.

For a single session without changing any settings:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=opas-local claude --model auto
```

## Credentials

opas forwards requests to Anthropic with one of two credentials:

- **opas's key (recommended).** The opas config names the key's variable
  (`api_key_env = "ANTHROPIC_API_KEY"`, as in the example config), and `ANTHROPIC_API_KEY`
  is set where opas runs. Claude Code sends the placeholder token, which opas drops.
- **Your own login or key (`--auth own`).** Remove `api_key_env` from the models in the
  opas config; opas then forwards Claude Code's `x-api-key` or `Authorization` header
  and its `anthropic-beta` header unchanged. With a claude.ai subscription login this
  works, but Claude Code's documentation names a gateway credential as the intended
  setup for gateways, and your subscription's limits still apply.

opas has no client authentication of its own: keep it on `127.0.0.1` (the default) or
behind an authenticating proxy, because anyone who can reach it can use its key.

## What you see

Claude Code shows the configured model, `auto`, not the model opas chose for each
request; it does not display response headers. To see the choices:

- opas logs every decision (`decided … model=…`) at the default log level;
- every response carries `x-opas-model`, the prediction (`x-opas-p-success`,
  `x-opas-cost-usd`, `x-opas-response-time-s`) and `x-opas-decision-id`;
- `GET /v1/decisions/{id}` shows, after the response, the actual tokens, cost and time
  next to the prediction, and the log line `completed` has the same;
- `POST /v1/route` explains a decision without sending anything upstream
  ([API](api.md)).

## Tuning

- **The balance between quality, speed and cost** is set in the opas config
  (`[router] weights`, default 60 / 20 / 20). To use another balance for one session,
  send a header:

  ```sh
  ANTHROPIC_CUSTOM_HEADERS="x-opas-preference: cost" claude --model auto
  ```

  Presets are `quality`, `balanced`, `cost` and `speed`; or
  `x-opas-weights: quality=80,speed=10,cost=10`.
- **Background requests** (session titles, summaries) use Claude Code's small-model
  setting, not `auto`. Point them straight at a cheap model with
  `"ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-5-5"` in `env`.
- **Context window:** Claude Code assumes 200K tokens for a model name it does not know.
  If your smallest candidate model has less, set `CLAUDE_CODE_MAX_CONTEXT_TOKENS`.

## Limits

- Claude Code's `/context` uses a character estimate, because opas does not implement
  the token-counting endpoint yet.
- Remote Control is not available while the base URL points away from Anthropic.
- The Claude desktop app ignores `ANTHROPIC_BASE_URL`; this setup is for the `claude`
  CLI.

## Undo

```sh
curl -fsSL https://raw.githubusercontent.com/panuhorsmalahti/opas/main/scripts/claude-code.sh | bash -s -- --uninstall
```

or delete the keys above from `~/.claude/settings.json`.
