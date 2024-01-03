cmake_minimum_required(VERSION 3.0)
project(recon_kernel)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(YAKL yakl)

add_executable(recon_kernel recon_kernel.cpp)
include(YAKL/yakl_utils.cmake)
yakl_process_target(recon_kernel)
target_link_libraries(recon_kernel yakl "${PORTURB_LINK_FLAGS}")
target_include_directories(recon_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)


