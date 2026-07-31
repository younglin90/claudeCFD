#!/usr/bin/env python3
"""Headless-browser E2E test of the electrospray GUI (real DOM + JS).

Starts the GUI server, drives it with a real Chromium via Playwright, and
asserts: page loads with no JS errors, every schema field renders as an
editable input/select, Save->Load round-trips through the form, Run Case
produces a summary table + history chart, and mesh generation fills the
patch table. Saves a screenshot.
"""
from __future__ import annotations
import json, subprocess, sys, time, urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8792
BASE = f"http://127.0.0.1:{PORT}"
CHROME = str(Path.home()/".cache/ms-playwright/chromium-1208/chrome-linux64/chrome")
SHOT = ROOT/"results"/"gui_e2e_screenshot.png"

results = []
def rec(name, ok, detail=""):
    results.append((name, ok, detail)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

def wait_ready():
    for _ in range(40):
        try:
            urllib.request.urlopen(BASE+"/api/ui-config", timeout=2); return True
        except Exception: time.sleep(0.5)
    return False

def main():
    srv = subprocess.Popen([sys.executable, str(ROOT/"apps"/"electrospray_gui_server.py"), "--port", str(PORT)],
                           cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        if not wait_ready():
            print("SERVER NOT UP"); print(srv.stdout.read() if srv.stdout else ""); return 2
        cfg = json.loads(urllib.request.urlopen(BASE+"/api/ui-config", timeout=5).read())
        schema = cfg["schema"]
        all_fields = [f for b in schema for f in b["fields"]]
        check_ids = {f["id"] for f in all_fields if f["kind"]=="check"}
        select_ids = {f["id"] for f in all_fields if f["kind"]=="select"}
        n_expected = len(all_fields)

        console_errors, page_errors = [], []
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
            pg = b.new_page(viewport={"width":1500,"height":1600})
            pg.on("console", lambda m: console_errors.append(m.text) if m.type=="error" else None)
            pg.on("pageerror", lambda e: page_errors.append(str(e)))
            pg.goto(BASE+"/", wait_until="networkidle")
            pg.wait_for_timeout(600)

            # 1. no JS errors on load
            rec("page load: no JS errors", not page_errors and not console_errors,
                f"pageerrors={page_errors[:2]} console={console_errors[:2]}")

            # 2. every schema field renders as the right editable control
            audit = pg.evaluate("""(args) => {
                const {check_ids, select_ids, ids} = args;
                const out = {missing:[], wrongtag:[], disabled:[], notEditable:[]};
                for (const id of ids) {
                    const el = document.getElementById(id);
                    if (!el) { out.missing.push(id); continue; }
                    const tag = el.tagName;
                    if (check_ids.includes(id)) {
                        if (!(tag==='INPUT' && el.type==='checkbox')) out.wrongtag.push(id+':'+tag);
                    } else if (select_ids.includes(id)) {
                        if (tag!=='SELECT') out.wrongtag.push(id+':'+tag);
                    } else {
                        if (tag!=='INPUT') out.wrongtag.push(id+':'+tag);
                    }
                    if (el.disabled) out.disabled.push(id);
                    if ((tag==='INPUT' && el.type!=='checkbox') && el.readOnly) out.notEditable.push(id);
                }
                return out;
            }""", {"check_ids": list(check_ids), "select_ids": list(select_ids),
                    "ids": [f["id"] for f in all_fields]})
            ok2 = not (audit["missing"] or audit["wrongtag"])
            rec(f"all {n_expected} fields render as editable controls", ok2,
                f"missing={audit['missing']} wrongtag={audit['wrongtag']} disabled={audit['disabled']} readonly={audit['notEditable']}")

            # 3. count actual rendered inputs/selects inside the field containers
            counts = pg.evaluate("""(containers) => {
                let inputs=0, selects=0, checks=0;
                for (const c of containers) {
                    const el = document.getElementById(c);
                    if (!el) continue;
                    inputs += el.querySelectorAll('input:not([type=checkbox])').length;
                    checks += el.querySelectorAll('input[type=checkbox]').length;
                    selects += el.querySelectorAll('select').length;
                }
                return {inputs, selects, checks};
            }""", [b["container"] for b in schema])
            total_rendered = counts["inputs"]+counts["selects"]+counts["checks"]
            rec("rendered control count matches schema", total_rendered==n_expected,
                f"rendered={total_rendered} expected={n_expected} detail={counts}")

            # 4. edit a numeric field and a checkbox (editable in practice)
            pg.fill("#target_ca_e", "0.29")
            pg.fill("#steps", "2")
            pg.fill("#nx", "6"); pg.fill("#ny", "10"); pg.fill("#nz", "6")
            pg.fill("#case_name", "gui_e2e_probe")
            pg.check("#use_moving_collector_wall")
            edited = pg.evaluate("""() => ({
                caE: document.getElementById('target_ca_e').value,
                moving: document.getElementById('use_moving_collector_wall').checked
            })""")
            rec("edit numeric + checkbox", edited["caE"]=="0.29" and edited["moving"]==True, f"{edited}")

            # 5. Save Case -> status pass, appears in dropdown after refresh
            pg.click("#saveCase")
            pg.wait_for_selector("#status.pass", timeout=15000)
            pg.wait_for_timeout(500)
            saved_ok = pg.evaluate("""() => {
                const sel=document.getElementById('savedCaseSelect');
                return [...sel.options].some(o=>o.value==='gui_e2e_probe');
            }""")
            rec("Save Case -> listed in dropdown", saved_ok, f"status={pg.text_content('#status')}")

            # 6. Mutate form, then Load Case restores saved values
            pg.fill("#target_ca_e", "0.99")
            pg.evaluate("""() => { const s=document.getElementById('savedCaseSelect'); s.value='gui_e2e_probe'; }""")
            pg.click("#loadCaseBtn")
            pg.wait_for_timeout(800)
            reloaded = pg.evaluate("() => document.getElementById('target_ca_e').value")
            rec("Load Case -> restores form", reloaded=="0.29", f"caE after load={reloaded}")

            # 7. Run Case -> summary table + history chart canvas visible
            pg.click("#runCase")
            pg.wait_for_selector("#status.pass, #status.error", timeout=120000)
            pg.wait_for_timeout(800)
            run_state = pg.evaluate("""() => ({
                status: document.getElementById('status').textContent,
                summaryRows: document.querySelectorAll('#resultsSummary table tr').length,
                chartVisible: document.getElementById('historyCanvas').style.display !== 'none',
                outputDir: document.getElementById('outputDir').textContent,
            })""")
            rec("Run Case -> pass + summary + chart", run_state["status"]=="pass" and run_state["summaryRows"]>0 and run_state["chartVisible"],
                f"{run_state}")

            # 8. Generate Nozzle Mesh -> patch rows fill
            pg.click("#generateNozzleMesh")
            pg.wait_for_selector("#status.pass, #status.error", timeout=60000)
            pg.wait_for_timeout(800)
            mesh_state = pg.evaluate("""() => ({
                status: document.getElementById('status').textContent,
                patchRows: document.querySelectorAll('#patchRows tr').length,
                meshMode: document.getElementById('mesh_mode').value,
            })""")
            rec("Generate Nozzle Mesh -> patch table filled", mesh_state["status"]=="pass" and mesh_state["patchRows"]>=4,
                f"{mesh_state}")

            # 9. patch BC editing: each patch row must expose role + BC selects
            bc_state = pg.evaluate("""() => {
                const rows=[...document.querySelectorAll('#patchRows tr')];
                if(!rows.length) return {rows:0};
                const first=rows[0];
                return {rows:rows.length,
                        selects:first.querySelectorAll('select').length,
                        numbers:first.querySelectorAll('input[type=number]').length};
            }""")
            rec("patch row exposes role+BC controls", bc_state.get("rows",0)>=4 and bc_state.get("selects",0)>=5,
                f"{bc_state}")

            SHOT.parent.mkdir(parents=True, exist_ok=True)
            pg.screenshot(path=str(SHOT), full_page=True)
            rec("screenshot saved", SHOT.exists(), str(SHOT))
            b.close()

        print("\n===== SUMMARY =====")
        npass=sum(1 for _,ok,_ in results if ok)
        print(f"{npass}/{len(results)} passed")
        for n,ok,d in results:
            if not ok: print(f"  FAIL: {n} :: {d}")
        return 0 if npass==len(results) else 1
    finally:
        srv.terminate()
        try: srv.wait(timeout=5)
        except Exception: srv.kill()

if __name__ == "__main__":
    sys.exit(main())
