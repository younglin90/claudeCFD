cmake_minimum_required(VERSION 3.20)
project(solver_electrospray_fvm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(OpenMP QUIET COMPONENTS CXX)

add_library(fvm_core INTERFACE)
target_include_directories(fvm_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fvm_core INTERFACE Eigen3::Eigen)
target_compile_options(fvm_core INTERFACE -O2 -Wall -Wextra -Wpedantic)
target_compile_definitions(fvm_core INTERFACE FVM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
  target_link_libraries(fvm_core INTERFACE OpenMP::OpenMP_CXX)
  message(STATUS "fvm_core: OpenMP enabled (${OpenMP_CXX_FLAGS})")
else()
  message(STATUS "fvm_core: OpenMP not found - building single-threaded")
endif()

add_executable(electrospray_validation_cli apps/electrospray_validation_cli.cpp)
target_link_libraries(electrospray_validation_cli PRIVATE fvm_core)

add_executable(verify_electrospray_cpp apps/verify_electrospray.cpp)
target_link_libraries(verify_electrospray_cpp PRIVATE fvm_core)

add_executable(defect_campaign_summary apps/defect_campaign_summary.cpp)
target_link_libraries(defect_campaign_summary PRIVATE fvm_core)

add_executable(electrospray_artifact_generator apps/electrospray_artifact_generator.cpp)
target_link_libraries(electrospray_artifact_generator PRIVATE fvm_core)

add_executable(electrospray_profile apps/electrospray_profile.cpp)
target_link_libraries(electrospray_profile PRIVATE fvm_core)

add_executable(electrospray_mesh_validate apps/electrospray_mesh_validate.cpp)
target_link_libraries(electrospray_mesh_validate PRIVATE fvm_core)

add_executable(electrospray_case_runner apps/electrospray_case_runner.cpp)
target_link_libraries(electrospray_case_runner PRIVATE fvm_core)

enable_testing()

function(add_fvm_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE fvm_core)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

add_fvm_test(test_mesh_geometry tests/test_mesh_geometry.cpp)
add_fvm_test(test_mesh3d_geometry tests/test_mesh3d_geometry.cpp)
add_fvm_test(test_mesh3d_tetra_geometry tests/test_mesh3d_tetra_geometry.cpp)
add_fvm_test(test_mesh3d_polyhedron_input tests/test_mesh3d_polyhedron_input.cpp)
add_fvm_test(test_openfoam_polymesh_reader3d tests/test_openfoam_polymesh_reader3d.cpp)
add_fvm_test(test_openfoam_field_reader3d tests/test_openfoam_field_reader3d.cpp)
add_fvm_test(test_gradients tests/test_gradients.cpp)
add_fvm_test(test_convection_tvd tests/test_convection_tvd.cpp)
add_fvm_test(test_convection_tvd3d tests/test_convection_tvd3d.cpp)
add_fvm_test(test_laplacian_mms tests/test_laplacian_mms.cpp)
add_fvm_test(test_laplacian3d_mms tests/test_laplacian3d_mms.cpp)
add_fvm_test(test_taylor_green tests/test_taylor_green.cpp)
add_fvm_test(test_taylor_green3d tests/test_taylor_green3d.cpp)
add_fvm_test(test_cavity_benchmarks tests/test_cavity_benchmarks.cpp)
add_fvm_test(test_cavity3d_smoke tests/test_cavity3d_smoke.cpp)
add_fvm_test(test_cavity3d_profiles tests/test_cavity3d_profiles.cpp)
add_fvm_test(test_cavity3d_reference_diagnostic tests/test_cavity3d_reference_diagnostic.cpp)
add_fvm_test(test_cavity3d_continuation_diagnostic tests/test_cavity3d_continuation_diagnostic.cpp)
add_fvm_test(test_cavity3d_resolution_sweep tests/test_cavity3d_resolution_sweep.cpp)
add_fvm_test(test_cavity3d_courant_reference_sweep tests/test_cavity3d_courant_reference_sweep.cpp)
add_fvm_test(test_vof_transport3d tests/test_vof_transport3d.cpp)
add_fvm_test(test_vof_isoadvector3d tests/test_vof_isoadvector3d.cpp)
add_fvm_test(test_plic_plane_cut3d tests/test_plic_plane_cut3d.cpp)
add_fvm_test(test_vof_shape3d tests/test_vof_shape3d.cpp)
add_fvm_test(test_surface_tension3d tests/test_surface_tension3d.cpp)
add_fvm_test(test_static_droplet3d tests/test_static_droplet3d.cpp)
add_fvm_test(test_static_droplet_adversarial3d tests/test_static_droplet_adversarial3d.cpp)
add_fvm_test(test_electrostatics3d tests/test_electrostatics3d.cpp)
add_fvm_test(test_ehd_coupling3d tests/test_ehd_coupling3d.cpp)
add_fvm_test(test_collocated_cavity_smoke tests/test_collocated_cavity_smoke.cpp)
add_fvm_test(test_collocated_re1000_diagnostic tests/test_collocated_re1000_diagnostic.cpp)
add_fvm_test(test_pressure_velocity tests/test_pressure_velocity.cpp)
add_fvm_test(test_pressure_velocity3d tests/test_pressure_velocity3d.cpp)
add_fvm_test(test_adversarial_meshes tests/test_adversarial_meshes.cpp)
add_fvm_test(test_benchmark_ledger tests/test_benchmark_ledger.cpp)
add_fvm_test(test_terminal_goal_ledger tests/test_terminal_goal_ledger.cpp)
add_fvm_test(test_hardening_audit tests/test_hardening_audit.cpp)
add_fvm_test(test_dynamic_droplet_oscillation3d tests/test_dynamic_droplet_oscillation3d.cpp)
add_fvm_test(test_linear_solver_diagnostics tests/test_linear_solver_diagnostics.cpp)
add_fvm_test(test_mesh3d_quality tests/test_mesh3d_quality.cpp)
add_fvm_test(test_vof_curvature_hardening3d tests/test_vof_curvature_hardening3d.cpp)
add_fvm_test(test_electrospray_scalar_models tests/test_electrospray_scalar_models.cpp)
add_fvm_test(test_electrospray_diagnostics tests/test_electrospray_diagnostics.cpp)
add_fvm_test(test_axisymmetric_electrostatics tests/test_axisymmetric_electrostatics.cpp)
add_fvm_test(test_electrospray_application_models tests/test_electrospray_application_models.cpp)
add_fvm_test(test_candido_cone_jet_smoke3d tests/test_candido_cone_jet_smoke3d.cpp)
add_fvm_test(test_candido_named_patch_boundary3d tests/test_candido_named_patch_boundary3d.cpp)
add_fvm_test(test_candido_conservative_surface_charge_closure3d tests/test_candido_conservative_surface_charge_closure3d.cpp)
add_fvm_test(test_candido_electrode_surface_current_boundary_isolation3d tests/test_candido_electrode_surface_current_boundary_isolation3d.cpp)
add_fvm_test(test_candido_interfacial_ohmic_charge_source3d tests/test_candido_interfacial_ohmic_charge_source3d.cpp)
add_fvm_test(test_candido_momentum_source_factorization3d tests/test_candido_momentum_source_factorization3d.cpp)
add_fvm_test(test_candido_long_window_budget3d tests/test_candido_long_window_budget3d.cpp)
add_fvm_test(test_candido_charge_closure_candidates3d tests/test_candido_charge_closure_candidates3d.cpp)
add_test(NAME test_electrospray_validation_cli_json
  COMMAND electrospray_validation_cli --format json)
add_test(NAME test_electrospray_validation_cli_markdown
  COMMAND electrospray_validation_cli --format markdown)
add_test(NAME test_electrospray_validation_cli_artifact_status
  COMMAND electrospray_validation_cli --format artifact-status-json)
add_test(NAME test_electrospray_validation_cli_check_all_artifacts
  COMMAND electrospray_validation_cli --check-all-artifacts)
add_test(NAME test_electrospray_validation_cli_reduced_step_json
  COMMAND electrospray_validation_cli --format reduced-step-json)
add_test(NAME test_defect_campaign_summary_cpp
  COMMAND defect_campaign_summary)
set_tests_properties(test_defect_campaign_summary_cpp PROPERTIES
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_electrospray_artifact_generator_cpp
  COMMAND electrospray_artifact_generator --output-dir ${CMAKE_CURRENT_BINARY_DIR}/generated_artifacts)
set_tests_properties(test_electrospray_artifact_generator_cpp PROPERTIES
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_electrospray_profile_smoke
  COMMAND electrospray_profile --output ${CMAKE_CURRENT_BINARY_DIR}/benchmark_logs/electrospray_profile.csv)
set_tests_properties(test_electrospray_profile_smoke PROPERTIES
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_electrospray_mesh_validate_builtin
  COMMAND electrospray_mesh_validate --builtin-hex --nx 3 --ny 4 --nz 2 --skew 0.03
          --output ${CMAKE_CURRENT_BINARY_DIR}/benchmark_logs/electrospray_mesh_validate_builtin.csv)
add_test(NAME test_electrospray_case_runner_gui
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_electrospray_case_runner_gui.py
          $<TARGET_FILE:electrospray_case_runner>
          $<TARGET_FILE:electrospray_mesh_validate>
          ${CMAKE_CURRENT_BINARY_DIR}/benchmark_logs/electrospray_case_runner_gui)
add_test(NAME test_openfoam_case_runner
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_openfoam_case_runner.py
          $<TARGET_FILE:electrospray_case_runner>
          ${CMAKE_CURRENT_BINARY_DIR}/benchmark_logs/openfoam_case_runner)
add_test(NAME test_nozzle_cad_generator
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nozzle_cad_generator.py
          ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/benchmark_logs/nozzle_cad_generator)
add_test(NAME test_electrospray_paper_gap_metric
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_electrospray_paper_gap_metric.py)
set_tests_properties(test_electrospray_paper_gap_metric PROPERTIES
  ENVIRONMENT "FVM_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME test_late_morphology_external_dataset_check
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/apps/electrospray_late_morphology_dataset_check.py --self-test)
add_test(NAME test_late_morphology_request_package_check
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/apps/electrospray_late_morphology_request_package_check.py)
set_tests_properties(test_benchmark_ledger PROPERTIES
  DEPENDS "test_cavity_benchmarks;test_collocated_re1000_diagnostic;test_taylor_green;test_pressure_velocity")
set_tests_properties(test_terminal_goal_ledger PROPERTIES
  DEPENDS "test_mesh3d_geometry;test_mesh3d_tetra_geometry;test_mesh3d_polyhedron_input;test_convection_tvd3d;test_laplacian3d_mms;test_pressure_velocity3d;test_cavity3d_profiles;test_cavity3d_reference_diagnostic;test_cavity3d_continuation_diagnostic;test_cavity3d_resolution_sweep;test_cavity3d_courant_reference_sweep;test_vof_transport3d;test_vof_isoadvector3d;test_vof_shape3d;test_surface_tension3d;test_static_droplet3d;test_static_droplet_adversarial3d;test_electrostatics3d;test_ehd_coupling3d")
set_tests_properties(test_hardening_audit PROPERTIES
  DEPENDS "test_terminal_goal_ledger;test_taylor_green3d;test_vof_shape3d;test_static_droplet_adversarial3d;test_ehd_coupling3d")
set_tests_properties(test_dynamic_droplet_oscillation3d PROPERTIES
  DEPENDS "test_surface_tension3d;test_static_droplet_adversarial3d;test_vof_transport3d;test_vof_isoadvector3d;test_pressure_velocity3d")
set_tests_properties(test_vof_curvature_hardening3d PROPERTIES
  DEPENDS "test_surface_tension3d;test_vof_transport3d;test_vof_isoadvector3d")
