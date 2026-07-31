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

add_executable(denner1d_run apps/denner1d_run.cpp)
target_link_libraries(denner1d_run PRIVATE denner1d)

add_executable(denner1d_dump apps/denner1d_dump.cpp)
target_link_libraries(denner1d_dump PRIVATE denner1d)

add_executable(denner1d_validate apps/denner1d_validate.cpp)
target_link_libraries(denner1d_validate PRIVATE denner1d)

add_executable(denner1d_unit tests/denner1d_unit.cpp)
target_link_libraries(denner1d_unit PRIVATE denner1d)
