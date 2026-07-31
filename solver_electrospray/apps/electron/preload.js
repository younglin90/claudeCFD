// Preload runs with contextIsolation enabled. The renderer loads the existing
// web GUI (served by the Python backend over http://127.0.0.1) which is a normal
// same-origin page using fetch() — it needs no privileged Node/Electron APIs, so
// this preload intentionally exposes nothing. Keeping it empty preserves the
// security boundary (nodeIntegration is off).
