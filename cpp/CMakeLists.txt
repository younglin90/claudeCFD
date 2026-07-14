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
