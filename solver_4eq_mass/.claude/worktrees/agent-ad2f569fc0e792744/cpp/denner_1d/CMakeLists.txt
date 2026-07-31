add_library(denner1d
    src/acid.cpp
    src/cases.cpp
    src/eos.cpp
    src/numerics.cpp
    src/png.cpp
    src/solver.cpp
    src/validation.cpp
)

target_include_directories(denner1d PUBLIC include)
target_compile_features(denner1d PUBLIC cxx_std_17)
target_compile_options(denner1d PRIVATE -Wall -Wextra -Wpedantic)
# (-march=native gives ~7% but only via FMA contraction, which breaks case01's machine-exact
#  PE gate; with -ffp-contract=off the gain vanishes -- the EOS is sqrt/div/branch heavy and
#  does not auto-vectorize. So no extra flags; the real win is OpenMP below.)

# OpenMP: the coupled Newton's per-cell/per-face kernels (compute_R, called 15x/iter for the
# FD Jacobian) are embarrassingly parallel; parallelize them across cores (machine has 64).
find_package(OpenMP REQUIRED)
target_link_libraries(denner1d PUBLIC OpenMP::OpenMP_CXX)

add_executable(denner1d_run apps/denner1d_run.cpp)
target_link_libraries(denner1d_run PRIVATE denner1d)

add_executable(denner1d_dump apps/denner1d_dump.cpp)
target_link_libraries(denner1d_dump PRIVATE denner1d)

add_executable(denner1d_validate apps/denner1d_validate.cpp)
target_link_libraries(denner1d_validate PRIVATE denner1d)

add_executable(denner1d_unit tests/denner1d_unit.cpp)
target_link_libraries(denner1d_unit PRIVATE denner1d)
