# Denner 1D C++ Port

This directory is the C++ replacement path for the former Python Denner 1D
workspace code.  The legacy Python sources were archived at the workspace root
as:

```text
legacy_python_sources_20260617.tar.gz
```

The archived file list is:

```text
cpp/denner_1d/python_sources_manifest.txt
```

## Build

```bash
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp -j 8
```

## Run

```bash
./build-cpp/cpp/denner_1d/denner1d_run 01
./build-cpp/cpp/denner_1d/denner1d_validate --out results_cpp/1D
```

The validation CLI writes comparison images to:

```text
results_cpp/1D/{case}/diff_vs_reference.png
```

## Current Numerical Status

The C++ path builds and runs without Python.  The current C++ solver is a first
replacement implementation of the EOS, limiters, MWI-style face velocity,
case setup, validation metrics, and PNG plotting.  It is not yet a full
line-for-line numerical port of the former implicit Python `assembly.py` and
`solver_a.py` operators.

Current `denner1d_validate` status after replacing the self-reference pass
with stricter profile/peak/HF-style gates:

```text
PASS: 01, 02, 04, 05
FAIL: 07, 13, 14, 15, 24, 25
```

Do not treat a validation result as meaningful if `solve_case()` returns
`reference_state()`.  That shortcut was removed; the current pass/fail status
comes from the actual C++ time-marching path.
