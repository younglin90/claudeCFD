#pragma once

// OpenMP parallel-for helper. Expands to the pragma only when compiled with
// -fopenmp (CMake links OpenMP::OpenMP_CXX onto fvm_core when found); otherwise it
// is a no-op, so builds stay warning-free under -Wall (no unknown-pragma warnings)
// on toolchains without OpenMP. Use only on loops whose iterations write disjoint
// outputs (e.g. one cell index per iteration) — never on face loops that scatter to
// both owner and neighbour, which would race.
#if defined(_OPENMP)
#define FVM_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define FVM_PARALLEL_FOR
#endif
