@echo off
rem Thin double-clickable wrapper -> delegates to run.ps1, which handles the WSL/UNC
rem repo layout, environment, and Electron-runtime bootstrap robustly.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
