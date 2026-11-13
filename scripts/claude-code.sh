#!/usr/bin/env bash
# Point Claude Code at a running opas router, by editing ~/.claude/settings.json.
#
#   curl -fsSL https://raw.githubusercontent.com/panuhorsmalahti/opas/main/scripts/claude-code.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/panuhorsmalahti/opas/main/scripts/claude-code.sh | bash -s -- --default
#
# Options:
#   --url URL        opas address (default: http://127.0.0.1:8080)
#   --default        also make `auto` the default model (otherwise pick it with /model auto
#                    or `claude --model auto`)
#   --auth opas      opas holds the provider API key (its config sets `api_key_env`); Claude
#                    Code gets a placeholder token so it does not ask you to log in (default)
#   --auth own       keep your own Claude Code login or API key; opas passes it through
#                    (the opas config must not set `api_key_env` for these models)
#   --settings PATH  settings file (default: $CLAUDE_CONFIG_DIR/settings.json or
#                    ~/.claude/settings.json)
#   --uninstall      remove what this script added
#   --dry-run        print the resulting settings instead of writing them
#
# A backup of the settings file is written next to it before any change.
set -euo pipefail

URL="http://127.0.0.1:8080"
AUTH="opas"
DEFAULT=0
UNINSTALL=0
DRY_RUN=0
SETTINGS="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/settings.json"
# Placeholder credential: opas replaces it with the provider key from its own config.
PLACEHOLDER="opas-local"

usage() { sed -n "2,22p" "$0" 2>/dev/null | sed "s/^# \{0,1\}//" || echo "See https://github.com/panuhorsmalahti/opas/blob/main/scripts/claude-code.sh"; }
fail() { printf 'opas claude-code: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="${2:?--url needs a value}"; shift 2 ;;
    --url=*) URL="${1#*=}"; shift ;;
    --auth) AUTH="${2:?--auth needs opas or own}"; shift 2 ;;
    --auth=*) AUTH="${1#*=}"; shift ;;
    --default) DEFAULT=1; shift ;;
    --settings) SETTINGS="${2:?--settings needs a path}"; shift 2 ;;
    --settings=*) SETTINGS="${1#*=}"; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1 (see --help)" ;;
  esac
done
case "$AUTH" in opas|own) ;; *) fail "--auth must be opas or own" ;; esac
URL="${URL%/}"

# JSON editing: jq if present, otherwise python3.
if command -v jq >/dev/null 2>&1; then
  edit() { jq --arg url "$URL" --arg auth "$AUTH" --argjson def "$DEFAULT" \
              --argjson un "$UNINSTALL" --arg ph "$PLACEHOLDER" "$JQ_PROGRAM"; }
elif command -v python3 >/dev/null 2>&1; then
  edit() { URL="$URL" AUTH="$AUTH" DEF="$DEFAULT" UN="$UNINSTALL" PH="$PLACEHOLDER" python3 -c "$PY_PROGRAM"; }
else
  fail "needs jq or python3 to edit $SETTINGS; or add the settings by hand (docs/usage/claude-code.md)"
fi

# shellcheck disable=SC2016  # jq/python code, expanded by jq/python, not the shell
JQ_PROGRAM='
  (.env // {}) as $env
  | if $un == 1 then
      .env = ($env
        | if .ANTHROPIC_BASE_URL == $url then del(.ANTHROPIC_BASE_URL) else . end
        | if .ANTHROPIC_AUTH_TOKEN == $ph then del(.ANTHROPIC_AUTH_TOKEN) else . end
        | if .ANTHROPIC_CUSTOM_MODEL_OPTION == "auto" then
            del(.ANTHROPIC_CUSTOM_MODEL_OPTION, .ANTHROPIC_CUSTOM_MODEL_OPTION_NAME,
                .ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION)
          else . end)
      | if .model == "auto" then del(.model) else . end
      | if .env == {} then del(.env) else . end
    else
      .env = ($env + {
          ANTHROPIC_BASE_URL: $url,
          ANTHROPIC_CUSTOM_MODEL_OPTION: "auto",
          ANTHROPIC_CUSTOM_MODEL_OPTION_NAME: "opas auto",
          ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION: "opas picks the model per request"
        }
        | if $auth == "opas" then .ANTHROPIC_AUTH_TOKEN = (.ANTHROPIC_AUTH_TOKEN // $ph)
          elif .ANTHROPIC_AUTH_TOKEN == $ph then del(.ANTHROPIC_AUTH_TOKEN)
          else . end)
      | if $def == 1 then .model = "auto" else . end
    end'

# shellcheck disable=SC2016  # jq/python code, expanded by jq/python, not the shell
PY_PROGRAM='
import json, os, sys
s = json.load(sys.stdin)
url, auth, ph = os.environ["URL"], os.environ["AUTH"], os.environ["PH"]
env = dict(s.get("env") or {})
if os.environ["UN"] == "1":
    if env.get("ANTHROPIC_BASE_URL") == url: env.pop("ANTHROPIC_BASE_URL")
    if env.get("ANTHROPIC_AUTH_TOKEN") == ph: env.pop("ANTHROPIC_AUTH_TOKEN")
    if env.get("ANTHROPIC_CUSTOM_MODEL_OPTION") == "auto":
        for k in ("ANTHROPIC_CUSTOM_MODEL_OPTION", "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME",
                  "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION"):
            env.pop(k, None)
    if s.get("model") == "auto": s.pop("model")
else:
    env.update({
        "ANTHROPIC_BASE_URL": url,
        "ANTHROPIC_CUSTOM_MODEL_OPTION": "auto",
        "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "opas auto",
        "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": "opas picks the model per request",
    })
    if auth == "opas":
        env.setdefault("ANTHROPIC_AUTH_TOKEN", ph)
    elif env.get("ANTHROPIC_AUTH_TOKEN") == ph:
        env.pop("ANTHROPIC_AUTH_TOKEN")
    if os.environ["DEF"] == "1": s["model"] = "auto"
if env: s["env"] = env
else: s.pop("env", None)
json.dump(s, sys.stdout, indent=2)
print()
'

current="{}"
if [ -s "$SETTINGS" ]; then
  current=$(cat "$SETTINGS")
fi
new=$(printf '%s' "$current" | edit) || fail "$SETTINGS is not valid JSON; fix it or edit by hand"

if [ "$DRY_RUN" = 1 ]; then
  printf '%s\n' "$new"
  exit 0
fi

mkdir -p "$(dirname "$SETTINGS")"
if [ -e "$SETTINGS" ]; then
  backup="$SETTINGS.bak-opas-$(date +%Y%m%d%H%M%S)"
  cp "$SETTINGS" "$backup"
fi
tmp="$SETTINGS.tmp-opas.$$"
printf '%s\n' "$new" > "$tmp"
mv "$tmp" "$SETTINGS"

if [ "$UNINSTALL" = 1 ]; then
  echo "Removed the opas settings from $SETTINGS${backup:+ (backup: $backup)}."
  exit 0
fi

echo "Claude Code now sends requests to opas at $URL"
echo "  settings: $SETTINGS${backup:+ (backup: $backup)}"
if [ "$AUTH" = opas ]; then
  echo "  auth:     opas uses the provider key from its own config (e.g. ANTHROPIC_API_KEY)"
else
  echo "  auth:     your own Claude Code login or API key, passed through by opas"
fi
if command -v curl >/dev/null 2>&1 && ! curl -fsS --max-time 2 "$URL/healthz" >/dev/null 2>&1; then
  echo ""
  echo "opas is not answering at $URL. Start it first:  opas"
fi
echo ""
if [ "$DEFAULT" = 1 ]; then
  echo "Start Claude Code as usual: \`claude\`. It uses the model \`auto\`."
else
  echo "Start Claude Code with \`claude --model auto\`, or pick \"opas auto\" in /model."
fi
echo "Undo: rerun this script with --uninstall."
