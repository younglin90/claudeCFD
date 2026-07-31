// Electron main process for the electrospray GUI.
//
// Architecture: this is a thin desktop shell. It reuses the *existing* Python
// backend (apps/electrospray_gui_server.py) and web frontend (apps/gui/index.html)
// unchanged. On startup it launches the server as a child process, waits until it
// is ready, then loads http://127.0.0.1:<port>/ in a native window. On Windows the
// server runs inside WSL (where the C++ binaries live); WSL2 forwards 127.0.0.1 to
// the host so the window can reach it. On Linux it runs python3 directly.

const { app, BrowserWindow, Menu, dialog } = require('electron');
const { spawn, execFile } = require('child_process');
const http = require('http');
const net = require('net');
const path = require('path');

const isWin = process.platform === 'win32';
const DISTRO = process.env.ESPRAY_WSL_DISTRO || 'ubuntu';
const SMOKE = process.env.ESPRAY_SMOKE === '1'; // headless self-test: verify then quit
const SHOT = process.env.ESPRAY_SHOT || '';     // if set (path), capture a screenshot then quit

let serverProc = null; // the spawned child (wsl.exe on Windows, python3 on Linux)
let serverPid = null;  // Linux PID of the python server (for a clean kill on Windows)
let serverPort = 0;
let win = null;

// --- path handling -------------------------------------------------------------

// Convert a Windows path to the equivalent WSL path so the server script can be
// addressed from inside WSL. Handles the \\wsl.localhost\<distro>\... and \\wsl$\...
// UNC forms (repo on the WSL filesystem) and drive paths (C:\ -> /mnt/c/).
function winPathToWsl(p) {
  p = p.replace(/\//g, '\\');
  let m = p.match(/^\\\\wsl(?:\.localhost|\$)\\[^\\]+\\(.*)$/i);
  if (m) return '/' + m[1].replace(/\\/g, '/');
  m = p.match(/^([A-Za-z]):\\(.*)$/);
  if (m) return '/mnt/' + m[1].toLowerCase() + '/' + m[2].replace(/\\/g, '/');
  return p.replace(/\\/g, '/');
}

function repoRoot() { return path.resolve(__dirname, '..', '..'); } // apps/electron -> repo
function serverScript() {
  const winPath = path.join(repoRoot(), 'apps', 'electrospray_gui_server.py');
  return isWin ? winPathToWsl(winPath) : winPath;
}

// --- server lifecycle ----------------------------------------------------------

function findFreePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.on('error', reject);
    s.listen(0, '127.0.0.1', () => {
      const p = s.address().port;
      s.close(() => resolve(p));
    });
  });
}

function startServer(port) {
  const script = serverScript();
  if (isWin) {
    // `exec` replaces the login shell with python, so the PID printed by `$$`
    // is the python server's own PID -> lets us kill it cleanly on quit.
    const bash = `echo "ESPRAY_PID=$$"; exec python3 '${script}' --host 127.0.0.1 --port ${port}`;
    serverProc = spawn('wsl.exe', ['-d', DISTRO, 'bash', '-lc', bash], { windowsHide: true });
  } else {
    serverProc = spawn('python3', [script, '--host', '127.0.0.1', '--port', String(port)]);
  }
  serverProc.stdout.on('data', (d) => {
    const s = d.toString();
    const m = s.match(/ESPRAY_PID=(\d+)/);
    if (m) serverPid = parseInt(m[1], 10);
    process.stdout.write('[server] ' + s);
  });
  serverProc.stderr.on('data', (d) => process.stderr.write('[server] ' + d.toString()));
  serverProc.on('exit', (code) => process.stdout.write(`[server] exited (${code})\n`));
}

let stopped = false;
function stopServer() {
  if (stopped) return;
  stopped = true;
  try {
    if (isWin && serverPid) {
      execFile('wsl.exe', ['-d', DISTRO, 'kill', '-TERM', String(serverPid)], () => {});
      setTimeout(() => execFile('wsl.exe', ['-d', DISTRO, 'kill', '-9', String(serverPid)], () => {}), 1500);
    } else if (serverProc && !isWin) {
      serverProc.kill('SIGTERM');
    }
  } catch (e) { /* best-effort */ }
  if (serverProc) { try { serverProc.kill(); } catch (e) { /* ignore */ } }
}

function waitForServer(port, timeoutMs = 45000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const req = http.get({ host: '127.0.0.1', port, path: '/api/ui-config', timeout: 2000 }, (res) => {
        res.resume();
        if (res.statusCode === 200) resolve();
        else retry();
      });
      req.on('error', retry);
      req.on('timeout', () => { req.destroy(); retry(); });
    };
    const retry = () => {
      if (Date.now() - start > timeoutMs) reject(new Error('backend did not become ready within ' + (timeoutMs / 1000) + 's'));
      else setTimeout(attempt, 500);
    };
    attempt();
  });
}

// --- window --------------------------------------------------------------------

async function createWindow() {
  win = new BrowserWindow({
    width: 1500,
    height: 1000,
    minWidth: 900,
    minHeight: 640,
    backgroundColor: '#f5f6f8',
    title: 'Electrospray 3D Runner',
    x: SHOT ? -4000 : undefined, // off-screen when only capturing a screenshot
    y: SHOT ? -4000 : undefined,
    show: !SMOKE || !!SHOT,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  await win.loadFile(path.join(__dirname, 'loading.html'));

  try {
    serverPort = await findFreePort();
    startServer(serverPort);
    await waitForServer(serverPort);
    await win.loadURL(`http://127.0.0.1:${serverPort}/`);
  } catch (err) {
    const msg = String((err && err.stack) || err);
    if (SMOKE) { process.stdout.write('SMOKE_FAIL ' + msg + '\n'); app.exit(2); return; }
    dialog.showErrorBox(
      'Electrospray GUI failed to start',
      msg + '\n\nCheck that:\n' +
      ` - the WSL distro "${DISTRO}" is available (override with ESPRAY_WSL_DISTRO)\n` +
      ' - the C++ binaries are built:  cmake --build build\n' +
      ' - python3 is on PATH inside WSL'
    );
    return;
  }

  if (SMOKE) {
    // Headless self-test: confirm the real page loaded AND its dynamic form finished
    // building (init() fetches /api/ui-config, then populates the field containers),
    // then quit. This exercises the full shell -> backend -> renderer path.
    try {
      let formControls = 0;
      for (let i = 0; i < 60; i++) {
        formControls = await win.webContents.executeJavaScript(
          'document.querySelectorAll("#fields-geometry input, #fields-charge-flags input, #fields-material input").length'
        );
        if (formControls >= 20) break;
        await new Promise((r) => setTimeout(r, 250));
      }
      const total = await win.webContents.executeJavaScript('document.querySelectorAll("input, select").length');
      const title = await win.webContents.executeJavaScript('document.querySelector("h1") && document.querySelector("h1").textContent');
      const ok = formControls >= 20 && title === 'Electrospray 3D Runner';
      process.stdout.write(`${ok ? 'SMOKE_OK' : 'SMOKE_FAIL'} form_controls=${formControls} total_controls=${total} title=${JSON.stringify(title)} port=${serverPort}\n`);
      if (SHOT && ok) {
        await new Promise((r) => setTimeout(r, 400));
        const img = await win.webContents.capturePage();
        if (!img.isEmpty()) {
          require('fs').writeFileSync(SHOT, img.toPNG());
          process.stdout.write('SHOT_SAVED ' + SHOT + '\n');
        } else {
          process.stdout.write('SHOT_EMPTY (no GPU paint; DOM render already verified)\n');
        }
      }
      if (!ok) { stopServer(); app.exit(4); return; }
    } catch (e) {
      process.stdout.write('SMOKE_FAIL executeJavaScript ' + String(e) + '\n');
      stopServer();
      app.exit(3);
      return;
    }
    stopServer();
    setTimeout(() => app.quit(), 500);
  }

  win.on('closed', () => { win = null; });
}

// --- app lifecycle -------------------------------------------------------------

app.whenReady().then(() => {
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: 'App', submenu: [{ role: 'reload' }, { role: 'forceReload' }, { role: 'toggleDevTools' }, { type: 'separator' }, { role: 'quit' }] },
    { label: 'View', submenu: [{ role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { type: 'separator' }, { role: 'togglefullscreen' }] },
  ]));
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on('window-all-closed', () => { stopServer(); if (process.platform !== 'darwin') app.quit(); });
app.on('before-quit', stopServer);
app.on('will-quit', stopServer);
process.on('exit', stopServer);
