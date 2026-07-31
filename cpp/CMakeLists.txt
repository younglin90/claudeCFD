cmake_minimum_required(VERSION 3.16)
project(claudeCFD_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -funroll-loops")

# ── Parallel backend ─────────────────────────────────────────────────────
# CPU multithread: OpenMP (g++ / nvc++). GPGPU offload: OpenACC (nvc++).
# Toggle with -DCFD_GPU=ON (requires nvc++ from the NVIDIA HPC SDK).
option(CFD_GPU "Enable OpenACC GPU offload (needs nvc++)" OFF)

add_library(cfd_flags INTERFACE)
target_include_directories(cfd_flags INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CFD_GPU)
  # NVIDIA HPC SDK: -acc=gpu enables OpenACC; -mp adds OpenMP.
  target_compile_options(cfd_flags INTERFACE -acc=gpu -gpu=cc89 -mp)
  target_link_options(cfd_flags INTERFACE -acc=gpu -gpu=cc89 -mp)
else()
  find_package(OpenMP REQUIRED)
  target_link_libraries(cfd_flags INTERFACE OpenMP::OpenMP_CXX)
endif()

add_executable(leveque_compare apps/leveque_compare.cpp)
target_link_libraries(leveque_compare PRIVATE cfd_flags)

add_executable(leveque_bench apps/leveque_bench.cpp)
target_link_libraries(leveque_bench PRIVATE cfd_flags)

add_executable(mach3_bench apps/mach3_bench.cpp)
target_link_libraries(mach3_bench PRIVATE cfd_flags)

add_executable(candflag_pp apps/candflag_pp.cpp)
target_link_libraries(candflag_pp PRIVATE cfd_flags)

add_executable(double_mach_bench apps/double_mach_bench.cpp)
target_link_libraries(double_mach_bench PRIVATE cfd_flags)

add_executable(double_mach_ramp_bench apps/double_mach_ramp_bench.cpp)
target_link_libraries(double_mach_ramp_bench PRIVATE cfd_flags)

add_executable(vortex_bench apps/vortex_bench.cpp)
target_link_libraries(vortex_bench PRIVATE cfd_flags)

add_executable(gresho_bench apps/gresho_bench.cpp)
target_link_libraries(gresho_bench PRIVATE cfd_flags)

add_executable(kh_linear_bench apps/kh_linear_bench.cpp)
target_link_libraries(kh_linear_bench PRIVATE cfd_flags)

add_executable(config3_bench apps/config3_bench.cpp)
target_link_libraries(config3_bench PRIVATE cfd_flags)

add_executable(shock_mixing_bench apps/shock_mixing_bench.cpp)
target_link_libraries(shock_mixing_bench PRIVATE cfd_flags)

add_executable(rt_bench apps/rt_bench.cpp)
target_link_libraries(rt_bench PRIVATE cfd_flags)

add_executable(kh_bench apps/kh_bench.cpp)
target_link_libraries(kh_bench PRIVATE cfd_flags)

add_executable(bench_euler2d apps/bench_euler2d.cpp)
target_link_libraries(bench_euler2d PRIVATE cfd_flags)

add_executable(euler3d_bench apps/euler3d_bench.cpp)
target_link_libraries(euler3d_bench PRIVATE cfd_flags)

add_executable(deform3d_bench apps/deform3d_bench.cpp)
target_link_libraries(deform3d_bench PRIVATE cfd_flags)

add_executable(rmi3d_bench apps/rmi3d_bench.cpp)
target_link_libraries(rmi3d_bench PRIVATE cfd_flags)

add_executable(shock_tube3d_bench apps/shock_tube3d_bench.cpp)
target_link_libraries(shock_tube3d_bench PRIVATE cfd_flags)

add_executable(langseth3d_bench apps/langseth3d_bench.cpp)
target_link_libraries(langseth3d_bench PRIVATE cfd_flags)

add_executable(octant3d_bench apps/octant3d_bench.cpp)
target_link_libraries(octant3d_bench PRIVATE cfd_flags)

add_executable(octant3d_unstr_bench apps/octant3d_unstr_bench.cpp)
target_link_libraries(octant3d_unstr_bench PRIVATE cfd_flags)

add_executable(langseth3d_unstr_bench apps/langseth3d_unstr_bench.cpp)
target_link_libraries(langseth3d_unstr_bench PRIVATE cfd_flags)

add_executable(deform3d_unstr_bench apps/deform3d_unstr_bench.cpp)
target_link_libraries(deform3d_unstr_bench PRIVATE cfd_flags)

add_executable(viscshock3d_unstr_bench apps/viscshock3d_unstr_bench.cpp)
target_link_libraries(viscshock3d_unstr_bench PRIVATE cfd_flags)

add_executable(validation_smoke apps/validation_smoke.cpp)
target_link_libraries(validation_smoke PRIVATE cfd_flags)

add_executable(thinc_cellshape_bench apps/thinc_cellshape_bench.cpp)
target_link_libraries(thinc_cellshape_bench PRIVATE cfd_flags)

add_executable(thinc_cellshape2d_bench apps/thinc_cellshape2d_bench.cpp)
target_link_libraries(thinc_cellshape2d_bench PRIVATE cfd_flags)

add_executable(tanh_quad_probe apps/tanh_quad_probe.cpp)
target_link_libraries(tanh_quad_probe PRIVATE cfd_flags)

add_executable(five_eq_02a_validate apps/five_eq_02a_validate.cpp)
target_link_libraries(five_eq_02a_validate PRIVATE cfd_flags)

add_executable(five_eq_07b_validate apps/five_eq_07b_validate.cpp)
target_link_libraries(five_eq_07b_validate PRIVATE cfd_flags)

add_executable(five_eq_profile_from_spec apps/five_eq_profile_from_spec.cpp)
target_link_libraries(five_eq_profile_from_spec PRIVATE cfd_flags)

add_executable(test_unstructured3d tests/test_unstructured3d.cpp)
target_link_libraries(test_unstructured3d PRIVATE cfd_flags)

add_executable(test_recon3d_unstr tests/test_recon3d_unstr.cpp)
target_link_libraries(test_recon3d_unstr PRIVATE cfd_flags)

add_executable(test_visc_grad tests/test_visc_grad.cpp)
target_link_libraries(test_visc_grad PRIVATE cfd_flags)

add_executable(test_mms_skew tests/test_mms_skew.cpp)
target_link_libraries(test_mms_skew PRIVATE cfd_flags)

enable_testing()
add_executable(test_eos tests/test_eos.cpp)
target_link_libraries(test_eos PRIVATE cfd_flags)
add_test(NAME test_eos COMMAND test_eos)

add_executable(test_primitive tests/test_primitive.cpp)
target_link_libraries(test_primitive PRIVATE cfd_flags)
add_test(NAME test_primitive COMMAND test_primitive)

add_executable(test_mesh tests/test_mesh.cpp)
target_link_libraries(test_mesh PRIVATE cfd_flags)
add_test(NAME test_mesh COMMAND test_mesh)

add_executable(test_mesh2d tests/test_mesh2d.cpp)
target_link_libraries(test_mesh2d PRIVATE cfd_flags)
add_test(NAME test_mesh2d COMMAND test_mesh2d)

add_executable(test_mesh_unstructured tests/test_mesh_unstructured.cpp)
target_link_libraries(test_mesh_unstructured PRIVATE cfd_flags)
target_compile_definitions(test_mesh_unstructured PRIVATE
  CC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/mesh_cc_ref.txt")
add_test(NAME test_mesh_unstructured COMMAND test_mesh_unstructured)

add_executable(test_adv2d tests/test_adv2d.cpp)
target_link_libraries(test_adv2d PRIVATE cfd_flags)
target_compile_definitions(test_adv2d PRIVATE
  ADV2D_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/adv2d_ref.txt")
add_test(NAME test_adv2d COMMAND test_adv2d)

add_executable(test_recon2d tests/test_recon2d.cpp)
target_link_libraries(test_recon2d PRIVATE cfd_flags)
target_compile_definitions(test_recon2d PRIVATE
  RECON_MLPU1_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/recon_mlpu1_ref.txt"
  RECON_TMLPU_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/recon_mlpu1tmlpu_ref.txt")
add_test(NAME test_recon2d COMMAND test_recon2d)

add_executable(test_rot2d tests/test_rot2d.cpp)
target_link_libraries(test_rot2d PRIVATE cfd_flags)
target_compile_definitions(test_rot2d PRIVATE
  ROT_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/rot_mlpu1_ref.txt")
add_test(NAME test_rot2d COMMAND test_rot2d)

add_executable(test_euler2d tests/test_euler2d.cpp)
target_link_libraries(test_euler2d PRIVATE cfd_flags)
target_compile_definitions(test_euler2d PRIVATE
  EUL2D_FO_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/eul2d_fo_ref.txt"
  EUL2D_MLPU1_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/eul2d_mlpu1_ref.txt"
  EUL2D_HLLC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/eul2d_hllc_ref.txt"
  EUL2D_REFL_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/eul2d_refl_ref.txt")
add_test(NAME test_euler2d COMMAND test_euler2d)

add_executable(test_sod1d tests/test_sod1d.cpp)
target_link_libraries(test_sod1d PRIVATE cfd_flags)
target_compile_definitions(test_sod1d PRIVATE
  SOD_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/sod1d_ref.txt"
  SOD_REF_MINMOD="${CMAKE_CURRENT_SOURCE_DIR}/tests/sod1d_ref_minmod.txt")
add_test(NAME test_sod1d COMMAND test_sod1d)

add_executable(test_recon_o2 tests/test_recon_o2.cpp)
target_link_libraries(test_recon_o2 PRIVATE cfd_flags)
add_test(NAME test_recon_o2 COMMAND test_recon_o2)

add_executable(recon3d_o2_test apps/recon3d_o2_test.cpp)
target_link_libraries(recon3d_o2_test PRIVATE cfd_flags)
add_test(NAME recon3d_o2_test COMMAND recon3d_o2_test)

# ── 5-equation IMEX production-path ports (Phase B, cfd/five_eq/) ─────────────
add_executable(test_5eq_sound_speed tests/test_5eq_sound_speed.cpp)
target_link_libraries(test_5eq_sound_speed PRIVATE cfd_flags)
target_compile_definitions(test_5eq_sound_speed PRIVATE
  SS_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/sound_speed_ref.txt")
add_test(NAME test_5eq_sound_speed COMMAND test_5eq_sound_speed)

add_executable(test_5eq_primitive_near_pure tests/test_5eq_primitive_near_pure.cpp)
target_link_libraries(test_5eq_primitive_near_pure PRIVATE cfd_flags)
target_compile_definitions(test_5eq_primitive_near_pure PRIVATE
  PRIMITIVE_NEAR_PURE_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/primitive_near_pure_ref.txt")
add_test(NAME test_5eq_primitive_near_pure COMMAND test_5eq_primitive_near_pure)

add_executable(test_5eq_weno tests/test_5eq_weno.cpp)
target_link_libraries(test_5eq_weno PRIVATE cfd_flags)
target_compile_definitions(test_5eq_weno PRIVATE
  WENO_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/weno5_face_ref.txt")
add_test(NAME test_5eq_weno COMMAND test_5eq_weno)

add_executable(test_5eq_slau2 tests/test_5eq_slau2.cpp)
target_link_libraries(test_5eq_slau2 PRIVATE cfd_flags)
target_compile_definitions(test_5eq_slau2 PRIVATE
  SLAU2_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/slau2_faces_ref.txt")
add_test(NAME test_5eq_slau2 COMMAND test_5eq_slau2)

add_executable(test_5eq_reconstruct tests/test_5eq_reconstruct.cpp)
target_link_libraries(test_5eq_reconstruct PRIVATE cfd_flags)
target_compile_definitions(test_5eq_reconstruct PRIVATE
  RECON_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/reconstruct_ref.txt")
add_test(NAME test_5eq_reconstruct COMMAND test_5eq_reconstruct)

add_executable(test_5eq_alpha_bvd tests/test_5eq_alpha_bvd.cpp)
target_link_libraries(test_5eq_alpha_bvd PRIVATE cfd_flags)
target_compile_definitions(test_5eq_alpha_bvd PRIVATE
  ALPHA_BVD_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/alpha_bvd_ref.txt")
add_test(NAME test_5eq_alpha_bvd COMMAND test_5eq_alpha_bvd)

add_executable(test_5eq_acoustic_solve tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_solve PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_solve PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_solve_ref.txt")
add_test(NAME test_5eq_acoustic_solve COMMAND test_5eq_acoustic_solve)

add_executable(test_5eq_acoustic_frozen tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_frozen PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_frozen PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_frozen_ref.txt"
  ACOUSTIC_FROZEN_MIXTURE=1)
add_test(NAME test_5eq_acoustic_frozen COMMAND test_5eq_acoustic_frozen)

add_executable(test_5eq_acoustic_interface_be tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_interface_be PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_interface_be PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_interface_be_ref.txt"
  ACOUSTIC_INTERFACE_BE=1)
add_test(NAME test_5eq_acoustic_interface_be COMMAND test_5eq_acoustic_interface_be)

add_executable(test_5eq_acoustic_pure_tol tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_pure_tol PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_pure_tol PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_pure_tol_ref.txt"
  ACOUSTIC_PURE_TOL_CONSISTENT=1)
add_test(NAME test_5eq_acoustic_pure_tol COMMAND test_5eq_acoustic_pure_tol)

add_executable(test_5eq_acoustic_acid tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_acid PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_acid PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_acid_ref.txt"
  ACOUSTIC_ACID=1)
add_test(NAME test_5eq_acoustic_acid COMMAND test_5eq_acoustic_acid)

add_executable(test_5eq_acoustic_trbdf2 tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_trbdf2 PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_trbdf2 PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_trbdf2_ref.txt"
  ACOUSTIC_TRBDF2=1)
add_test(NAME test_5eq_acoustic_trbdf2 COMMAND test_5eq_acoustic_trbdf2)

add_executable(test_5eq_acoustic_waf tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_waf PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_waf PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_waf_ref.txt"
  ACOUSTIC_WAF=1 ACOUSTIC_RECON_COMPONENT=1)
add_test(NAME test_5eq_acoustic_waf COMMAND test_5eq_acoustic_waf)

add_executable(test_5eq_acoustic_diss_consistent tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_diss_consistent PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_diss_consistent PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_diss_consistent_ref.txt"
  ACOUSTIC_DISS_CONSISTENT=1 ACOUSTIC_RECON_COMPONENT=1)
add_test(NAME test_5eq_acoustic_diss_consistent COMMAND test_5eq_acoustic_diss_consistent)

add_executable(test_5eq_acoustic_characteristic tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_characteristic PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_characteristic PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_characteristic_ref.txt"
  ACOUSTIC_RECON_MODE=2)
add_test(NAME test_5eq_acoustic_characteristic COMMAND test_5eq_acoustic_characteristic)

add_executable(test_5eq_acoustic_weno3 tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_weno3 PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_weno3 PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_weno3_ref.txt"
  ACOUSTIC_RECON_MODE=3)
add_test(NAME test_5eq_acoustic_weno3 COMMAND test_5eq_acoustic_weno3)

add_executable(test_5eq_acoustic_muscl3 tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_muscl3 PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_muscl3 PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_muscl3_ref.txt"
  ACOUSTIC_RECON_MODE=4)
add_test(NAME test_5eq_acoustic_muscl3 COMMAND test_5eq_acoustic_muscl3)

add_executable(test_5eq_acoustic_bvd tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_bvd PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_bvd PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_bvd_ref.txt"
  ACOUSTIC_RECON_MODE=5)
add_test(NAME test_5eq_acoustic_bvd COMMAND test_5eq_acoustic_bvd)

add_executable(test_5eq_acoustic_interface_one_sided tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_interface_one_sided PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_interface_one_sided PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_interface_one_sided_ref.txt"
  ACOUSTIC_RECON_COMPONENT=1 ACOUSTIC_INTERFACE_CENTERED=0)
add_test(NAME test_5eq_acoustic_interface_one_sided COMMAND test_5eq_acoustic_interface_one_sided)

add_executable(test_5eq_acoustic_first_order tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_first_order PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_first_order PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_first_order_ref.txt"
  ACOUSTIC_RECON_COMPONENT=1 ACOUSTIC_MUSCL=0)
add_test(NAME test_5eq_acoustic_first_order COMMAND test_5eq_acoustic_first_order)

add_executable(test_5eq_acoustic_stencil_clean tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_stencil_clean PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_stencil_clean PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_stencil_clean_ref.txt"
  ACOUSTIC_RECON_COMPONENT=1 ACOUSTIC_STENCIL_CLEAN=1)
add_test(NAME test_5eq_acoustic_stencil_clean COMMAND test_5eq_acoustic_stencil_clean)

add_executable(test_5eq_acoustic_waf_one_minus_nu tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_waf_one_minus_nu PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_waf_one_minus_nu PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_waf_one_minus_nu_ref.txt"
  ACOUSTIC_RECON_COMPONENT=1 ACOUSTIC_WAF=1 ACOUSTIC_WAF_SIGMA=1)
add_test(NAME test_5eq_acoustic_waf_one_minus_nu COMMAND test_5eq_acoustic_waf_one_minus_nu)

add_executable(test_5eq_acoustic_waf_sensor tests/test_5eq_acoustic_solve.cpp)
target_link_libraries(test_5eq_acoustic_waf_sensor PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_waf_sensor PRIVATE
  ACOUSTIC_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_waf_sensor_ref.txt"
  ACOUSTIC_RECON_COMPONENT=1 ACOUSTIC_WAF=1 ACOUSTIC_WAF_SIGMA=2)
add_test(NAME test_5eq_acoustic_waf_sensor COMMAND test_5eq_acoustic_waf_sensor)

add_executable(test_5eq_material_update tests/test_5eq_material_update.cpp)
target_link_libraries(test_5eq_material_update PRIVATE cfd_flags)
target_compile_definitions(test_5eq_material_update PRIVATE
  MATERIAL_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/material_update_ref.txt")
add_test(NAME test_5eq_material_update COMMAND test_5eq_material_update)

add_executable(test_5eq_eos_facade tests/test_5eq_eos_facade.cpp)
target_link_libraries(test_5eq_eos_facade PRIVATE cfd_flags)
add_test(NAME test_5eq_eos_facade COMMAND test_5eq_eos_facade)

add_executable(test_5eq_primitive_utils tests/test_5eq_primitive_utils.cpp)
target_link_libraries(test_5eq_primitive_utils PRIVATE cfd_flags)
add_test(NAME test_5eq_primitive_utils COMMAND test_5eq_primitive_utils)

add_executable(test_5eq_material_source_modes tests/test_5eq_material_source_modes.cpp)
target_link_libraries(test_5eq_material_source_modes PRIVATE cfd_flags)
target_compile_definitions(test_5eq_material_source_modes PRIVATE
  MATERIAL_SOURCE_MODES_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/material_source_modes_ref.txt")
add_test(NAME test_5eq_material_source_modes COMMAND test_5eq_material_source_modes)

add_executable(test_5eq_material_options tests/test_5eq_material_options.cpp)
target_link_libraries(test_5eq_material_options PRIVATE cfd_flags)
target_compile_definitions(test_5eq_material_options PRIVATE
  MATERIAL_OPTIONS_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/material_options_ref.txt")
add_test(NAME test_5eq_material_options COMMAND test_5eq_material_options)

add_executable(test_5eq_primitive_filters tests/test_5eq_primitive_filters.cpp)
target_link_libraries(test_5eq_primitive_filters PRIVATE cfd_flags)
target_compile_definitions(test_5eq_primitive_filters PRIVATE
  PRIMITIVE_FILTERS_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/primitive_filters_ref.txt")
add_test(NAME test_5eq_primitive_filters COMMAND test_5eq_primitive_filters)

add_executable(test_5eq_pure_euler tests/test_5eq_pure_euler.cpp)
target_link_libraries(test_5eq_pure_euler PRIVATE cfd_flags)
target_compile_definitions(test_5eq_pure_euler PRIVATE
  PURE_EULER_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/pure_euler_ref.txt")
add_test(NAME test_5eq_pure_euler COMMAND test_5eq_pure_euler)

add_executable(test_5eq_energy_flux tests/test_5eq_energy_flux.cpp)
target_link_libraries(test_5eq_energy_flux PRIVATE cfd_flags)
target_compile_definitions(test_5eq_energy_flux PRIVATE
  ENERGY_FLUX_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/energy_flux_ref.txt")
add_test(NAME test_5eq_energy_flux COMMAND test_5eq_energy_flux)

add_executable(test_5eq_step tests/test_5eq_step.cpp)
target_link_libraries(test_5eq_step PRIVATE cfd_flags)
target_compile_definitions(test_5eq_step PRIVATE
  STEP_02A_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/step_02A_ref.txt"
  STEP_07B_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/step_07B_ref.txt")
add_test(NAME test_5eq_step COMMAND test_5eq_step)

function(add_5eq_step_closure_test CLOSURE_NAME CLOSURE_VALUE)
  add_executable(test_5eq_step_${CLOSURE_NAME} tests/test_5eq_step.cpp)
  target_link_libraries(test_5eq_step_${CLOSURE_NAME} PRIVATE cfd_flags)
  target_compile_definitions(test_5eq_step_${CLOSURE_NAME} PRIVATE
    STEP_02A_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/step_02A_${CLOSURE_NAME}_ref.txt"
    STEP_07B_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/step_07B_${CLOSURE_NAME}_ref.txt"
    STEP_FORCE_CLOSURE=${CLOSURE_VALUE})
  add_test(NAME test_5eq_step_${CLOSURE_NAME} COMMAND test_5eq_step_${CLOSURE_NAME})
endfunction()
add_5eq_step_closure_test(implicit_energy 2)
add_5eq_step_closure_test(implicit_energy_momentum 3)
add_5eq_step_closure_test(apec_pe 7)
target_compile_definitions(test_5eq_step_implicit_energy_momentum PRIVATE STEP_TOL=1e-11)

add_executable(test_5eq_run tests/test_5eq_run.cpp)
target_link_libraries(test_5eq_run PRIVATE cfd_flags)
target_compile_definitions(test_5eq_run PRIVATE
  RUN_02A_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/run_02A_ref.txt"
  SSP3_TRANSPORT_02A_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/ssp3_transport_02A_ref.txt"
  RUN_07B_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/run_07B_ref.txt"
  SSP2_07B_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/ssp2_07B_ref.txt"
  SSP3_TRANSPORT_07B_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/ssp3_transport_07B_ref.txt")
add_test(NAME test_5eq_run COMMAND test_5eq_run)

add_executable(test_oscillation_guards tests/test_oscillation_guards.cpp)
target_link_libraries(test_oscillation_guards PRIVATE cfd_flags)
add_test(NAME test_oscillation_guards COMMAND test_oscillation_guards)

add_executable(test_5eq_source_terms tests/test_5eq_source_terms.cpp)
target_link_libraries(test_5eq_source_terms PRIVATE cfd_flags)
add_test(NAME test_5eq_source_terms COMMAND test_5eq_source_terms)

add_executable(test_5eq_source_terms_oracle tests/test_5eq_source_terms_oracle.cpp)
target_link_libraries(test_5eq_source_terms_oracle PRIVATE cfd_flags)
target_compile_definitions(test_5eq_source_terms_oracle PRIVATE
  SOURCE_TERMS_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/source_terms_ref.txt")
add_test(NAME test_5eq_source_terms_oracle COMMAND test_5eq_source_terms_oracle)

add_executable(test_5eq_boundary tests/test_5eq_boundary.cpp)
target_link_libraries(test_5eq_boundary PRIVATE cfd_flags)
add_test(NAME test_5eq_boundary COMMAND test_5eq_boundary)

add_executable(test_5eq_explicit tests/test_5eq_explicit.cpp)
target_link_libraries(test_5eq_explicit PRIVATE cfd_flags)
target_compile_definitions(test_5eq_explicit PRIVATE
  EXPLICIT_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/explicit_step_ref.txt")
add_test(NAME test_5eq_explicit COMMAND test_5eq_explicit)

add_executable(test_5eq_explicit_alpha_options tests/test_5eq_explicit_alpha_options.cpp)
target_link_libraries(test_5eq_explicit_alpha_options PRIVATE cfd_flags)
target_compile_definitions(test_5eq_explicit_alpha_options PRIVATE
  EXPLICIT_ALPHA_OPTIONS_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/explicit_alpha_options_ref.txt")
add_test(NAME test_5eq_explicit_alpha_options COMMAND test_5eq_explicit_alpha_options)

add_executable(test_5eq_relaxation tests/test_5eq_relaxation.cpp)
target_link_libraries(test_5eq_relaxation PRIVATE cfd_flags)
add_test(NAME test_5eq_relaxation COMMAND test_5eq_relaxation)

add_executable(test_5eq_primitive_nd tests/test_5eq_primitive_nd.cpp)
target_link_libraries(test_5eq_primitive_nd PRIVATE cfd_flags)
add_test(NAME test_5eq_primitive_nd COMMAND test_5eq_primitive_nd)

add_executable(test_5eq_implicit_faces tests/test_5eq_implicit_faces.cpp)
target_link_libraries(test_5eq_implicit_faces PRIVATE cfd_flags)
add_test(NAME test_5eq_implicit_faces COMMAND test_5eq_implicit_faces)

add_executable(test_5eq_explicit_residual tests/test_5eq_explicit_residual.cpp)
target_link_libraries(test_5eq_explicit_residual PRIVATE cfd_flags)
target_compile_definitions(test_5eq_explicit_residual PRIVATE
  EXPLICIT_POSITIVITY_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/explicit_residual_positivity_ref.txt")
add_test(NAME test_5eq_explicit_residual COMMAND test_5eq_explicit_residual)

add_executable(test_5eq_ars222 tests/test_5eq_ars222.cpp)
target_link_libraries(test_5eq_ars222 PRIVATE cfd_flags)
target_compile_definitions(test_5eq_ars222 PRIVATE ARS_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/ars222_step_ref.txt")
add_test(NAME test_5eq_ars222 COMMAND test_5eq_ars222)

add_executable(test_5eq_legacy_integrators tests/test_5eq_legacy_integrators.cpp)
target_link_libraries(test_5eq_legacy_integrators PRIVATE cfd_flags)
target_compile_definitions(test_5eq_legacy_integrators PRIVATE
  LEGACY_INTEGRATORS_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/legacy_integrators_ref.txt")
add_test(NAME test_5eq_legacy_integrators COMMAND test_5eq_legacy_integrators)

add_executable(test_5eq_python_solve_defaults tests/test_5eq_python_solve_defaults.cpp)
target_link_libraries(test_5eq_python_solve_defaults PRIVATE cfd_flags)
target_compile_definitions(test_5eq_python_solve_defaults PRIVATE
  PYTHON_SOLVE_DEFAULTS_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/python_solve_defaults_ref.txt")
add_test(NAME test_5eq_python_solve_defaults COMMAND test_5eq_python_solve_defaults)

add_executable(test_5eq_ssp3_stage_residual tests/test_5eq_ssp3_stage_residual.cpp)
target_link_libraries(test_5eq_ssp3_stage_residual PRIVATE cfd_flags)
target_compile_definitions(test_5eq_ssp3_stage_residual PRIVATE
  SSP3_STAGE_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/ssp3_stage_residual_ref.txt")
add_test(NAME test_5eq_ssp3_stage_residual COMMAND test_5eq_ssp3_stage_residual)

add_executable(test_5eq_ssp3_explicit_residual tests/test_5eq_ssp3_explicit_residual.cpp)
target_link_libraries(test_5eq_ssp3_explicit_residual PRIVATE cfd_flags)
target_compile_definitions(test_5eq_ssp3_explicit_residual PRIVATE
  SSP3_EXPLICIT_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/ssp3_stage_residual_explicit_ref.txt")
add_test(NAME test_5eq_ssp3_explicit_residual COMMAND test_5eq_ssp3_explicit_residual)

add_executable(test_5eq_pe_correction tests/test_5eq_pe_correction.cpp)
target_link_libraries(test_5eq_pe_correction PRIVATE cfd_flags)
target_compile_definitions(test_5eq_pe_correction PRIVATE PE_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/pe_correction_ref.txt")
add_test(NAME test_5eq_pe_correction COMMAND test_5eq_pe_correction)

add_executable(test_5eq_pe_projection_modes tests/test_5eq_pe_projection_modes.cpp)
target_link_libraries(test_5eq_pe_projection_modes PRIVATE cfd_flags)
target_compile_definitions(test_5eq_pe_projection_modes PRIVATE PE_MODES_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/pe_projection_modes_ref.txt")
add_test(NAME test_5eq_pe_projection_modes COMMAND test_5eq_pe_projection_modes)

add_executable(test_5eq_pe_diagnostic tests/test_5eq_pe_diagnostic.cpp)
target_link_libraries(test_5eq_pe_diagnostic PRIVATE cfd_flags)
target_compile_definitions(test_5eq_pe_diagnostic PRIVATE PE_DIAG_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/pe_diagnostic_ref.txt")
add_test(NAME test_5eq_pe_diagnostic COMMAND test_5eq_pe_diagnostic)

add_executable(test_5eq_helmholtz tests/test_5eq_helmholtz.cpp)
target_link_libraries(test_5eq_helmholtz PRIVATE cfd_flags)
target_compile_definitions(test_5eq_helmholtz PRIVATE HELMHOLTZ_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/helmholtz_ref.txt")
add_test(NAME test_5eq_helmholtz COMMAND test_5eq_helmholtz)
add_executable(test_5eq_schur_blocks tests/test_5eq_schur_blocks.cpp)
target_link_libraries(test_5eq_schur_blocks PRIVATE cfd_flags)
target_compile_definitions(test_5eq_schur_blocks PRIVATE SCHUR_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/schur_blocks_ref.txt")
add_test(NAME test_5eq_schur_blocks COMMAND test_5eq_schur_blocks)

add_executable(test_5eq_schur_iteration tests/test_5eq_schur_iteration.cpp)
target_link_libraries(test_5eq_schur_iteration PRIVATE cfd_flags)
target_compile_definitions(test_5eq_schur_iteration PRIVATE SCHUR_ITERATION_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/schur_iteration_ref.txt")
add_test(NAME test_5eq_schur_iteration COMMAND test_5eq_schur_iteration)

add_executable(test_5eq_rhie_chow tests/test_5eq_rhie_chow.cpp)
target_link_libraries(test_5eq_rhie_chow PRIVATE cfd_flags)
target_compile_definitions(test_5eq_rhie_chow PRIVATE RHIE_CHOW_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/rhie_chow_ref.txt")
add_test(NAME test_5eq_rhie_chow COMMAND test_5eq_rhie_chow)

add_executable(test_5eq_ars_rhie_chow tests/test_5eq_ars_rhie_chow.cpp)
target_link_libraries(test_5eq_ars_rhie_chow PRIVATE cfd_flags)
target_compile_definitions(test_5eq_ars_rhie_chow PRIVATE ARS_RHIE_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/ars222_rhie_chow_ref.txt")
add_test(NAME test_5eq_ars_rhie_chow COMMAND test_5eq_ars_rhie_chow)

add_executable(test_5eq_acoustic_riemann_faces tests/test_5eq_acoustic_riemann_faces.cpp)
target_link_libraries(test_5eq_acoustic_riemann_faces PRIVATE cfd_flags)
target_compile_definitions(test_5eq_acoustic_riemann_faces PRIVATE ACOUSTIC_RIEMANN_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/acoustic_riemann_faces_ref.txt")
add_test(NAME test_5eq_acoustic_riemann_faces COMMAND test_5eq_acoustic_riemann_faces)

add_executable(test_5eq_ars_acoustic_riemann tests/test_5eq_ars_acoustic_riemann.cpp)
target_link_libraries(test_5eq_ars_acoustic_riemann PRIVATE cfd_flags)
target_compile_definitions(test_5eq_ars_acoustic_riemann PRIVATE ARS_ACOUSTIC_RIEMANN_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/ars222_acoustic_riemann_ref.txt")
add_test(NAME test_5eq_ars_acoustic_riemann COMMAND test_5eq_ars_acoustic_riemann)

add_executable(test_5eq_upwind_implicit_faces tests/test_5eq_upwind_implicit_faces.cpp)
target_link_libraries(test_5eq_upwind_implicit_faces PRIVATE cfd_flags)
target_compile_definitions(test_5eq_upwind_implicit_faces PRIVATE UPWIND_FACE_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/upwind_implicit_faces_ref.txt")
add_test(NAME test_5eq_upwind_implicit_faces COMMAND test_5eq_upwind_implicit_faces)

add_executable(test_5eq_compact_implicit tests/test_5eq_compact_implicit.cpp)
target_link_libraries(test_5eq_compact_implicit PRIVATE cfd_flags)
target_compile_definitions(test_5eq_compact_implicit PRIVATE COMPACT_IMPLICIT_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/compact_implicit_ref.txt")
add_test(NAME test_5eq_compact_implicit COMMAND test_5eq_compact_implicit)

add_executable(test_5eq_positivity tests/test_5eq_positivity.cpp)
target_link_libraries(test_5eq_positivity PRIVATE cfd_flags)
target_compile_definitions(test_5eq_positivity PRIVATE POSITIVITY_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/positivity_theta_ref.txt")
add_test(NAME test_5eq_positivity COMMAND test_5eq_positivity)

add_executable(test_5eq_low_flux tests/test_5eq_low_flux.cpp)
target_link_libraries(test_5eq_low_flux PRIVATE cfd_flags)
target_compile_definitions(test_5eq_low_flux PRIVATE LOW_FLUX_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/low_flux_ref.txt")
add_test(NAME test_5eq_low_flux COMMAND test_5eq_low_flux)

add_executable(test_5eq_face_state_options tests/test_5eq_face_state_options.cpp)
target_link_libraries(test_5eq_face_state_options PRIVATE cfd_flags)
target_compile_definitions(test_5eq_face_state_options PRIVATE FACE_OPTIONS_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/face_state_options_ref.txt")
add_test(NAME test_5eq_face_state_options COMMAND test_5eq_face_state_options)

add_executable(test_5eq_nd_transport tests/test_5eq_nd_transport.cpp)
target_link_libraries(test_5eq_nd_transport PRIVATE cfd_flags)
add_test(NAME test_5eq_nd_transport COMMAND test_5eq_nd_transport)

add_executable(test_5eq_nd_reconstruction tests/test_5eq_nd_reconstruction.cpp)
target_link_libraries(test_5eq_nd_reconstruction PRIVATE cfd_flags)
add_test(NAME test_5eq_nd_reconstruction COMMAND test_5eq_nd_reconstruction)

add_executable(test_5eq_nd_solver tests/test_5eq_nd_solver.cpp)
target_link_libraries(test_5eq_nd_solver PRIVATE cfd_flags)
target_compile_definitions(test_5eq_nd_solver PRIVATE
  ND_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/nd2d_step_ref.txt"
  TMLPU_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/nd2d_tmlpu_step_ref.txt"
  BOUNDARY_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/nd2d_boundary_step_ref.txt"
  GRAVITY_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/nd2d_gravity_step_ref.txt")
add_test(NAME test_5eq_nd_solver COMMAND test_5eq_nd_solver)

add_executable(test_5eq_nd3_solver tests/test_5eq_nd3_solver.cpp)
target_link_libraries(test_5eq_nd3_solver PRIVATE cfd_flags)
target_compile_definitions(test_5eq_nd3_solver PRIVATE
  ND3_REF="${CMAKE_CURRENT_SOURCE_DIR}/tests/5eq_ref/nd3d_tmlpu_step_ref.txt")
add_test(NAME test_5eq_nd3_solver COMMAND test_5eq_nd3_solver)
