#!/usr/bin/env python3
"""End-to-end self-test for the electrospray GUI server.

Starts the server as a subprocess, exercises every HTTP endpoint the browser
uses, and reports PASS/FAIL per endpoint. Kills the server on exit.
Run: python3 apps/_gui_selftest.py
"""
from __future__ import annotations
import json, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8791
BASE = f"http://127.0.0.1:{PORT}"

results = []
def rec(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def post(path, body, timeout=300):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def main():
    srv = subprocess.Popen([sys.executable, str(ROOT/"apps"/"electrospray_gui_server.py"),
                            "--port", str(PORT)],
                           cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # wait for readiness
        ready = False
        for _ in range(30):
            try:
                get("/api/ui-config"); ready = True; break
            except Exception:
                time.sleep(0.5)
        if not ready:
            print("SERVER DID NOT START"); print(srv.stdout.read() if srv.stdout else ""); return 2

        # ---- GET endpoints ----
        st, cfg = get("/api/ui-config")
        blocks = [b["container"] for b in cfg.get("schema", [])]
        rec("GET /api/ui-config", st==200 and len(blocks)==11,
            f"status={st} blocks={len(blocks)}")

        st, dc = get("/api/default-case")
        # every schema field id must be present in default-case
        schema_ids = [f["id"] for b in cfg["schema"] for f in b["fields"]]
        missing = [i for i in schema_ids if i not in dc]
        rec("GET /api/default-case", st==200 and not missing,
            f"status={st} nfields={len(schema_ids)} missing={missing}")

        # index.html served
        try:
            with urllib.request.urlopen(BASE + "/", timeout=10) as r:
                html = r.read().decode(); html_ok = "Electrospray 3D Runner" in html
            rec("GET / (index.html)", html_ok, f"len={len(html)}")
        except Exception as e:
            rec("GET / (index.html)", False, str(e))

        # ---- SSOT drift check: default_case vs --print-defaults ----
        exe = ROOT/"build"/"electrospray_case_runner"
        pd = subprocess.run([str(exe), "--print-defaults"], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=30)
        drift = []
        if pd.returncode == 0:
            defs = json.loads(pd.stdout)
            for k, v in defs.items():
                if k in dc and dc[k] != v:
                    # tolerate float repr
                    try:
                        if abs(float(dc[k]) - float(v)) <= 1e-12 * max(1.0, abs(float(v))):
                            continue
                    except Exception:
                        pass
                    drift.append((k, dc[k], v))
            rec("SSOT default_case == runner --print-defaults", not drift,
                f"drift={drift}" if drift else f"{len(defs)} keys agree")
        else:
            rec("SSOT --print-defaults", False, pd.stderr[:200])

        # ---- POST mesh-preview (builtin) ----
        case = dict(dc); case["case_name"]="selftest_case"; case["mesh_mode"]="builtin_hex"
        st, mp = post("/api/mesh-preview", case)
        rec("POST /api/mesh-preview builtin", st==200 and mp.get("status")=="pass" and mp.get("faces"),
            f"status={mp.get('status')} faces={mp.get('preview_face_count')}")

        # ---- POST validate-mesh (builtin) ----
        st, vm = post("/api/validate-mesh", case)
        rec("POST /api/validate-mesh builtin", st==200 and vm.get("status")=="pass",
            f"status={vm.get('status')}")

        # ---- POST generate-nozzle-mesh ----
        gcase = dict(dc); gcase["case_name"]="selftest_nozzle"; gcase["nx"]=12; gcase["ny"]=20; gcase["nz"]=12
        st, gm = post("/api/generate-nozzle-mesh", gcase, timeout=120)
        patches = [p["name"] for p in gm.get("patches", [])] if isinstance(gm.get("patches"), list) else gm.get("patches")
        rec("POST /api/generate-nozzle-mesh", st==200 and gm.get("status")=="pass",
            f"status={gm.get('status')} cells={gm.get('cells')} patches={patches}")

        # ---- mesh-preview on generated openfoam polyMesh ----
        if gm.get("generated_polyMesh"):
            ocase = dict(dc); ocase["mesh_mode"]="openfoam_polyMesh"; ocase["openfoam_polyMesh"]=gm["generated_polyMesh"]
            st, omp = post("/api/mesh-preview", ocase)
            rec("POST /api/mesh-preview openfoam", st==200 and omp.get("status")=="pass",
                f"status={omp.get('status')} faces={omp.get('preview_face_count')}")

        # ---- save-case + list-cases + load-case roundtrip ----
        case["target_ca_e"]=0.31; case["steps"]=2; case["nx"]=6; case["ny"]=10; case["nz"]=6
        case["boundary_conditions"]={"nozzle_electrode":{"velocity":{"type":"noSlip","value":[0,0,0]},
            "pressure":{"type":"zeroGradient","value":0},"alpha":{"type":"zeroGradient","value":0},
            "potential":{"type":"fixedValue","value":2180.0},"charge":{"type":"zeroGradient","value":0}}}
        case["patch_roles"]={"nozzle_electrode":"electrode"}
        st, sv = post("/api/save-case", case)
        rec("POST /api/save-case", st==200 and sv.get("status")=="pass", f"status={sv.get('status')}")

        st, lc = get("/api/list-cases")
        rec("GET /api/list-cases", st==200 and "selftest_case" in lc.get("cases",[]),
            f"cases={lc.get('cases')}")

        st, loaded = get("/api/load-case?name=selftest_case")
        rt_ok = (loaded.get("target_ca_e")==0.31 and loaded.get("steps")==2 and
                 loaded.get("boundary_conditions",{}).get("nozzle_electrode",{}).get("potential",{}).get("value")==2180.0)
        rec("GET /api/load-case roundtrip", st==200 and rt_ok,
            f"caE={loaded.get('target_ca_e')} steps={loaded.get('steps')} bc_kept={rt_ok}")

        # ---- run-case smoke (tiny) ----
        run = dict(dc); run["case_name"]="selftest_run"; run["run_mode"]="candido_smoke"
        run["mesh_mode"]="builtin_hex"; run["nx"]=6; run["ny"]=10; run["nz"]=6; run["steps"]=2
        run["target_ca_e"]=0.25
        t0=time.time()
        st, rr = post("/api/run-case", run, timeout=600)
        dt=time.time()-t0
        has_hist = isinstance(rr.get("history"),dict) and rr["history"].get("rows")
        rec("POST /api/run-case smoke", st==200 and rr.get("status")=="pass",
            f"status={rr.get('status')} steps={rr.get('steps')} cells={rr.get('cells')} "
            f"mass_drift={rr.get('alpha_mass_drift')} hist_rows={len(rr['history']['rows']) if has_hist else 0} {dt:.1f}s")

        # ---- run-history after a run ----
        st, rh = get("/api/run-history?name=selftest_run")
        rec("GET /api/run-history", st==200 and rh.get("status")=="pass" and rh.get("history",{}).get("columns"),
            f"status={rh.get('status')} cols={rh.get('history',{}).get('columns')}")

        # ---- error handling: load nonexistent case ----
        st, err = get("/api/load-case?name=__nope__")
        rec("GET /api/load-case missing -> 404", st==404 and err.get("status")=="error",
            f"status_code={st}")

        print("\n===== SUMMARY =====")
        npass = sum(1 for _,ok,_ in results if ok)
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
