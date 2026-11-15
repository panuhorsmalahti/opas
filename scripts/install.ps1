# Install opas on Windows: the server binary, the trained predictor, the encoder model and
# a config.
#
#   irm https://raw.githubusercontent.com/panuhorsmalahti/opas/main/scripts/install.ps1 | iex
#
# Environment variables (all optional):
#   OPAS_VERSION     release tag to install, e.g. v0.1.1 (default: the latest release)
#   OPAS_BIN_DIR     where opas.exe and opas-encode.exe go (default: %LOCALAPPDATA%\opas\bin)
#   OPAS_HOME        predictor and encoder model (default: %LOCALAPPDATA%\opas)
#   OPAS_CONFIG_DIR  config directory (default: %APPDATA%\opas)
#   OPAS_SKIP_MODEL  set to 1 to skip the 9.5 GB encoder model download
#
# Supported: Windows x86_64 with AVX2.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Repo = 'panuhorsmalahti/opas'
$ModelUrl = 'https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf'
$ModelFile = 'Qwen3.5-9B-Q8_0.gguf'

function Fail($msg) { Write-Error "opas install: $msg"; exit 1 }

$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -ne 'AMD64') { Fail "no prebuilt binary for Windows $arch; use WSL with install.sh or build from source" }
$target = 'x86_64-pc-windows-msvc'

$BinDir = if ($env:OPAS_BIN_DIR) { $env:OPAS_BIN_DIR } else { Join-Path $env:LOCALAPPDATA 'opas\bin' }
$OpasHome = if ($env:OPAS_HOME) { $env:OPAS_HOME } else { Join-Path $env:LOCALAPPDATA 'opas' }
$ConfigDir = if ($env:OPAS_CONFIG_DIR) { $env:OPAS_CONFIG_DIR } else { Join-Path $env:APPDATA 'opas' }

# The latest release's tag, from the redirect of /releases/latest (no API rate limit).
$tag = $env:OPAS_VERSION
if (-not $tag) {
  $resp = Invoke-WebRequest -Uri "https://github.com/$Repo/releases/latest" -Method Head -MaximumRedirection 5 -UseBasicParsing
  $tag = ($resp.BaseResponse.ResponseUri, $resp.BaseResponse.RequestMessage.RequestUri | Where-Object { $_ } | Select-Object -First 1).Segments[-1]
  if (-not ($tag -like 'v*')) { Fail "no release found at https://github.com/$Repo/releases" }
}

$name = "opas-$tag-$target"
$base = "https://github.com/$Repo/releases/download/$tag"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("opas-install-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
  Write-Host "Installing opas $tag ($target)"
  $zip = Join-Path $tmp "$name.zip"
  Invoke-WebRequest -Uri "$base/$name.zip" -OutFile $zip -UseBasicParsing
  $sums = (Invoke-WebRequest -Uri "$base/SHA256SUMS" -UseBasicParsing).Content
  if ($sums -is [byte[]]) { $sums = [System.Text.Encoding]::UTF8.GetString($sums) }
  $line = ($sums -split "`n") | Where-Object { $_ -match "\s$([regex]::Escape("$name.zip"))\s*$" } | Select-Object -First 1
  if (-not $line) { Fail "$name.zip is not listed in SHA256SUMS" }
  $want = ($line -split '\s+')[0].ToLower()
  $have = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
  if ($want -ne $have) { Fail "checksum mismatch for $name.zip" }
  Expand-Archive -Path $zip -DestinationPath $tmp -Force
  $src = Join-Path $tmp $name

  New-Item -ItemType Directory -Force -Path $BinDir, (Join-Path $OpasHome 'probes'), (Join-Path $OpasHome 'models\gguf'), $ConfigDir | Out-Null
  Copy-Item (Join-Path $src 'opas.exe'), (Join-Path $src 'opas-encode.exe') $BinDir -Force
  Copy-Item (Join-Path $src 'probes\*') (Join-Path $OpasHome 'probes') -Force
  Write-Host "  binaries:  $BinDir\opas.exe, $BinDir\opas-encode.exe"
  Write-Host "  predictor: $OpasHome\probes\"

  # Config: the example, with absolute paths (forward slashes: valid in TOML strings and
  # accepted by Windows). An existing config is left alone.
  $config = Join-Path $ConfigDir 'opas.toml'
  $homeFwd = $OpasHome -replace '\\', '/'
  if (Test-Path $config) {
    Copy-Item (Join-Path $src 'opas.example.toml') (Join-Path $ConfigDir 'opas.example.toml') -Force
    Write-Host "  config:    $config (kept; the new example is at $ConfigDir\opas.example.toml)"
  } else {
    (Get-Content (Join-Path $src 'opas.example.toml')) `
      -replace '^encoder_model = "models/gguf/', "encoder_model = `"$homeFwd/models/gguf/" `
      -replace '^probe = "probes/', "probe = `"$homeFwd/probes/" |
      Set-Content -Encoding utf8 $config
    Write-Host "  config:    $config"
  }
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

$model = Join-Path $OpasHome "models\gguf\$ModelFile"
if ($env:OPAS_SKIP_MODEL -eq '1') {
  Write-Host "  encoder:   skipped (OPAS_SKIP_MODEL=1); the router needs $model"
} elseif ((Test-Path $model) -and -not (Test-Path "$model.part")) {
  Write-Host "  encoder:   $model (already present)"
} else {
  Write-Host "  encoder:   downloading $ModelFile (9.5 GB; resumes if interrupted)"
  # curl.exe ships with Windows 10 and later; -C - resumes a partial download.
  & curl.exe -fL --retry 5 -C - -o "$model.part" $ModelUrl
  if ($LASTEXITCODE -ne 0) { Fail 'model download failed; run the installer again to resume' }
  Move-Item -Force "$model.part" $model
  Write-Host "  encoder:   $model"
}

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not (($userPath -split ';') -contains $BinDir)) {
  [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ";$BinDir").TrimStart(';'), 'User')
  $env:Path = "$env:Path;$BinDir"
  Write-Host ""
  Write-Host "Added $BinDir to your user PATH (open a new terminal to pick it up)."
}

Write-Host ""
Write-Host "Edit models and keys in $config, then start the router:"
Write-Host ""
Write-Host '  $env:ANTHROPIC_API_KEY = "..."   # the key the config''s api_key_env names'
Write-Host '  opas'
Write-Host ""
Write-Host "It listens on http://127.0.0.1:8080 (OpenAI: /v1, Anthropic: the root URL)."
