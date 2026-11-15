#!/usr/bin/env bash
# Point OpenAI Codex at a running opas router.
#
#   curl -fsSL https://raw.githubusercontent.com/panuhorsmalahti/opas/main/scripts/codex.sh | bash
#   codex --profile opas
#
# By default this adds a Codex profile, ~/.codex/opas.config.toml, so only
# `codex --profile opas` goes through opas and ~/.codex/config.toml is not touched.
#
# Options:
#   --url URL           opas address (default: http://127.0.0.1:8080)
#   --default           route all Codex sessions through opas: add the settings to
#                       ~/.codex/config.toml instead (between `# >>> opas` markers)
#   --preference NAME   send `x-opas-preference: NAME` (quality, balanced, cost, speed)
#   --auth opas         opas holds the OpenAI key (its config sets `api_key_env`); Codex
#                       sends a placeholder token (default)
#   --auth own          send your own OPENAI_API_KEY through opas (the opas config must
#                       not set `api_key_env` for these models)
#   --codex-home DIR    Codex's directory (default: $CODEX_HOME or ~/.codex)
#   --uninstall         remove what this script added
#   --dry-run           print what would be written instead of writing it
#
# Codex then uses the model `auto` from provider `opas` (Responses API at URL/v1), with
# a model catalog (~/.codex/opas-catalog.json) that gives `auto` the tools and settings
# of a current GPT model, which Codex would not enable for an unknown model name. The
# opas config must route between OpenAI models; see docs/usage/codex.md.
set -euo pipefail

REPO_RAW="https://raw.githubusercontent.com/panuhorsmalahti/opas/main"
URL="http://127.0.0.1:8080"
AUTH="opas"
DEFAULT=0
PREFERENCE=""
UNINSTALL=0
DRY_RUN=0
CODEX_DIR="${CODEX_HOME:-$HOME/.codex}"

fail() { printf 'opas codex: %s\n' "$*" >&2; exit 1; }
usage() { sed -n "2,26p" "$0" 2>/dev/null | sed "s/^# \{0,1\}//" || echo "See $REPO_RAW/scripts/codex.sh"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="${2:?--url needs a value}"; shift 2 ;;
    --url=*) URL="${1#*=}"; shift ;;
    --default) DEFAULT=1; shift ;;
    --preference) PREFERENCE="${2:?--preference needs a value}"; shift 2 ;;
    --preference=*) PREFERENCE="${1#*=}"; shift ;;
    --auth) AUTH="${2:?--auth needs opas or own}"; shift 2 ;;
    --auth=*) AUTH="${1#*=}"; shift ;;
    --codex-home) CODEX_DIR="${2:?--codex-home needs a path}"; shift 2 ;;
    --codex-home=*) CODEX_DIR="${1#*=}"; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1 (see --help)" ;;
  esac
done
case "$AUTH" in opas|own) ;; *) fail "--auth must be opas or own" ;; esac
case "$PREFERENCE" in ""|quality|balanced|cost|speed) ;; *) fail "--preference must be quality, balanced, cost or speed" ;; esac
URL="${URL%/}"

CONFIG="$CODEX_DIR/config.toml"
PROFILE="$CODEX_DIR/opas.config.toml"
CATALOG="$CODEX_DIR/opas-catalog.json"
BEGIN="# >>> opas (added by opas scripts/codex.sh; remove with --uninstall) >>>"
END="# <<< opas <<<"
MARK="# Written by opas scripts/codex.sh; remove with --uninstall."

# `config.toml` without this script's blocks, blank lines trimmed at both ends.
config_without_opas() {
  [ -f "$CONFIG" ] || return 0
  awk -v b="$BEGIN" -v e="$END" '
    $0 == b { skip = 1; next }
    $0 == e { skip = 0; next }
    !skip { print }' "$CONFIG" | sed -e '/./,$!d' | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}'
}

top_level_keys() {
  printf '%s\n' "$1" | awk '/^[[:space:]]*\[/ { exit } { print }'
}

check_toml() {
  command -v python3 >/dev/null 2>&1 || return 0
  printf '%s\n' "$1" | python3 -c 'import sys
try:
    import tomllib
except ImportError:
    sys.exit(0)
tomllib.loads(sys.stdin.read())' || fail "the result would not be valid TOML; nothing was changed"
}

write_file() { # path content
  if [ -f "$1" ]; then
    backup="$1.bak-opas-$(date +%Y%m%d%H%M%S)"
    cp "$1" "$backup"
    echo "  backup:  $backup"
  fi
  printf '%s\n' "$2" > "$1.tmp-opas.$$"
  mv "$1.tmp-opas.$$" "$1"
}

owned_profile() { [ -f "$PROFILE" ] && head -1 "$PROFILE" | grep -qF "$MARK"; }

if [ "$UNINSTALL" = 1 ]; then
  if owned_profile; then
    if [ "$DRY_RUN" = 1 ]; then echo "would remove $PROFILE"; else rm -f "$PROFILE"; echo "Removed $PROFILE"; fi
  fi
  if [ -f "$CONFIG" ] && grep -qF "$BEGIN" "$CONFIG"; then
    rest=$(config_without_opas)
    if [ "$DRY_RUN" = 1 ]; then printf '%s\n' "$rest"; exit 0; fi
    write_file "$CONFIG" "$rest"
    echo "Removed the opas settings from $CONFIG"
  fi
  [ "$DRY_RUN" = 1 ] || rm -f "$CATALOG"
  exit 0
fi

if [ "$AUTH" = opas ]; then
  auth_line='experimental_bearer_token = "opas-local"   # placeholder: opas uses its own OpenAI key'
else
  auth_line='env_key = "OPENAI_API_KEY"                   # your key, passed through by opas'
fi
selection=$(
  echo 'model_provider = "opas"'
  echo 'model = "auto"'
  echo "model_catalog_json = \"$CATALOG\""
)
provider=$(
  echo "[model_providers.opas]"
  echo 'name = "opas"'
  echo "base_url = \"$URL/v1\""
  echo 'wire_api = "responses"'
  echo "$auth_line"
  if [ -n "$PREFERENCE" ]; then echo "http_headers = { \"x-opas-preference\" = \"$PREFERENCE\" }"; fi
  echo 'capabilities = { remote_compaction = "unsupported" }'
)

if [ "$DEFAULT" = 1 ]; then
  rest=$(config_without_opas)
  if top_level_keys "$rest" | grep -Eq '^[[:space:]]*(model|model_provider|model_catalog_json)[[:space:]]*='; then
    fail "$CONFIG already sets model, model_provider or model_catalog_json; remove them, or run without --default and use \`codex --profile opas\`"
  fi
  if printf '%s\n' "$rest" | grep -Eq '^\[model_providers\.opas\]'; then
    fail "$CONFIG already defines [model_providers.opas] outside this script's markers; remove it first"
  fi
  new="$BEGIN"$'\n'"$selection"$'\n'"$END"$'\n\n'
  if [ -n "$rest" ]; then new="$new$rest"$'\n\n'; fi
  new="$new$BEGIN"$'\n'"$provider"$'\n'"$END"
  target="$CONFIG"
else
  if [ -f "$PROFILE" ] && ! owned_profile; then
    fail "$PROFILE exists and was not written by this script; move it away first"
  fi
  new="$MARK"$'\n'"$selection"$'\n\n'"$provider"
  target="$PROFILE"
fi
check_toml "$new"

if [ "$DRY_RUN" = 1 ]; then
  echo "# $target"
  printf '%s\n' "$new"
  exit 0
fi

mkdir -p "$CODEX_DIR"
echo "Codex can now use opas at $URL"
write_file "$target" "$new"
echo "  config:  $target"
# Switching between modes: drop the other mode's settings.
if [ "$DEFAULT" = 1 ] && owned_profile; then rm -f "$PROFILE"; fi
if [ "$DEFAULT" = 0 ] && [ -f "$CONFIG" ] && grep -qF "$BEGIN" "$CONFIG"; then
  write_file "$CONFIG" "$(config_without_opas)"
fi

# The model catalog: from this checkout if run from one, else from GitHub.
here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -n "$here" ] && [ -f "$here/codex/opas-catalog.json" ]; then
  cp "$here/codex/opas-catalog.json" "$CATALOG"
else
  curl -fsSL -o "$CATALOG" "$REPO_RAW/scripts/codex/opas-catalog.json" || fail "could not download the model catalog"
fi
echo "  catalog: $CATALOG"

if command -v curl >/dev/null 2>&1 && ! curl -fsS --max-time 2 "$URL/healthz" >/dev/null 2>&1; then
  echo ""
  echo "opas is not answering at $URL. Start it first:  opas"
fi
echo ""
if [ "$DEFAULT" = 1 ]; then
  echo "Start Codex as usual: \`codex\`. It uses opas with the model \`auto\`."
else
  echo "Start Codex with opas:  codex --profile opas"
fi
echo "The opas config must route between OpenAI models (see docs/usage/codex.md)."
echo "Undo: rerun this script with --uninstall."
