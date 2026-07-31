# Electrospray GUI — Electron desktop app

A native desktop wrapper around the existing electrospray web GUI. It is a **thin
shell**: it reuses the Python backend (`apps/electrospray_gui_server.py`) and the web
frontend (`apps/gui/index.html`) unchanged, launching the server as a child process and
displaying it in a native window.

## How it works

1. On launch, `main.js` picks a free port and starts the backend.
   - **Windows host:** the server runs inside WSL (`wsl.exe -d ubuntu python3 …`), where
     the C++ solver binaries live. WSL2 forwards `127.0.0.1` to the host, so the window
     reaches it. The Linux PID is captured (via `exec` + `$$`) for a clean shutdown.
   - **Linux host:** `python3` is run directly.
2. It waits until `GET /api/ui-config` returns 200, then loads `http://127.0.0.1:<port>/`.
3. On quit, the backend is terminated.

No solver logic is duplicated in Node — every case run still goes through the validated
Python + C++ path.

## Prerequisites

- **Node.js** on the host (Windows: `node`/`npm` in PATH).
- The C++ binaries built: from the repo root, `cmake --build build` (inside WSL).
- On Windows: a WSL distro named `ubuntu` (override with the `ESPRAY_WSL_DISTRO` env var).

## Run

### Windows (repo on the WSL filesystem) — use the launcher

Double-click **`run.cmd`** (or `powershell -File run.ps1`). On first launch it installs a
local Electron runtime into `%LOCALAPPDATA%\electrospray-gui-runtime` (one time), then
opens the app. Subsequent launches are instant.

> Why a launcher instead of `npm install` + `npm start`? The repo lives on the WSL
> filesystem (`\\wsl.localhost\...`). Windows cannot use a UNC path as the working
> directory during Electron's postinstall, so a plain `npm install` inside `apps/electron`
> fails. The launcher installs the Electron runtime in a local folder and points it at the
> app in the repo — no solver code is moved or duplicated.

### Linux host (running inside WSL with a display), or repo on a local drive

```sh
cd apps/electron
npm install      # first time only — downloads Electron
npm start
```

## Headless self-test

Verifies the whole stack (spawn backend → window → real page + JS render) without
showing a window:

```sh
ESPRAY_SMOKE=1 npm start      # prints SMOKE_OK … and exits 0 on success
```

## Package a Windows installer (optional)

```sh
npm install --save-dev electron-builder
npm run dist                  # NSIS installer under dist/
```

## Environment variables

| Var | Default | Meaning |
| --- | --- | --- |
| `ESPRAY_WSL_DISTRO` | `ubuntu` | WSL distro that hosts the backend (Windows only). |
| `ESPRAY_SMOKE` | unset | `1` = run the headless self-test and quit. |

## Notes

- The port is chosen dynamically to avoid collisions; nothing is hard-coded.
- If the window shows the error dialog, the backend could not start — confirm the WSL
  distro name, that `python3` is on PATH inside WSL, and that `build/` contains the
  compiled `electrospray_case_runner`.
