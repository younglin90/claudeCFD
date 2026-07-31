# Steady-State LBM Local Web GUI

This GUI is a thin local wrapper around the existing benchmark drivers.  It
does not modify solver code.  Runs are launched as subprocesses and written to
`web_gui_runs/<job-id>/`.

## Start

```bash
python3 web_gui/gui_server.py --host 127.0.0.1 --port 8767
```

Open:

```text
http://127.0.0.1:8767
```

If that port is already in use, pass another free local port and open the same
address with the new port number.

## What It Controls

- Benchmark runner:
  - `run_ap_schur_proposed_only.py`
  - `paper_60case_benchmark_no_force_scaling.py`
- Mesh scaling levels: `1x`, `2x`, `3x`
- Case families:
  - channel, Couette, cavity Re100/400/1000, backward step, cylinder wake,
    multi-cylinder, T-junction
- Methods:
  - proposed, Picard, Anderson, preconditioned LBM, inexact Newton,
    dual-time MG
- Solver/runtime environment variables:
  - AP-Schur enable/disable, rectangular/mask-aware flags, Krylov settings,
    outer-loop limits, tail settings, max-step override, thread counts

The advanced JSON box shows the exact payload sent to the local API.  Edit it
when a parameter is not exposed by a dedicated widget.
