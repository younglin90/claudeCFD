# Windows launcher for the electrospray Electron app (PowerShell — robust with the
# repo living on the WSL filesystem \\wsl.localhost\...).
#
# `npm install electron` cannot run inside the repo's UNC app dir (Windows rejects a
# UNC path as the working directory during Electron's postinstall). So this script
# installs/caches the Electron runtime in a LOCAL folder and launches the app (which
# stays in the repo) from there. The app then starts the WSL Python backend + C++ solver.

$ErrorActionPreference = 'Stop'
$appDir  = $PSScriptRoot
$base    = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } elseif ($env:TEMP) { $env:TEMP } else { $HOME }
$rt      = Join-Path $base 'electrospray-gui-runtime'
$electron = Join-Path $rt 'node_modules\electron\dist\electron.exe'

Write-Host "[info] app     = $appDir"
Write-Host "[info] runtime = $rt"

if (-not (Test-Path $electron)) {
  Write-Host "[setup] Installing the Electron runtime once (first launch only)..."
  New-Item -ItemType Directory -Force -Path $rt | Out-Null
  $pkg = Join-Path $rt 'package.json'
  if (-not (Test-Path $pkg)) { '{ "name": "electrospray-gui-runtime", "private": true }' | Out-File -Encoding ascii $pkg }
  Push-Location $rt          # local CWD so npm's postinstall works
  try { & npm install electron@33 } finally { Pop-Location }
}

if (-not (Test-Path $electron)) {
  Write-Error "Electron runtime install failed. Ensure Node.js/npm are on PATH and you have network access, then re-run."
  exit 1
}

# Launch the app from a local CWD, passing the (UNC) app path explicitly.
Set-Location -LiteralPath $base
& $electron "$appDir\." @args
exit $LASTEXITCODE
