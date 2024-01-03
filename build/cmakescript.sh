#!/bin/bash

./cmakeclean.sh

cmake      \
  -DYAKL_CUDA_FLAGS="${YAKL_CUDA_FLAGS}"         \
  -DYAKL_CXX_FLAGS="${YAKL_CXX_FLAGS}"           \
  -DYAKL_SYCL_FLAGS="${YAKL_SYCL_FLAGS}"         \
  -DYAKL_OPENMP_FLAGS="${YAKL_OPENMP_FLAGS}"     \
  -DYAKL_HIP_FLAGS="${YAKL_HIP_FLAGS}"           \
  -DYAKL_F90_FLAGS="${YAKL_F90_FLAGS}"           \
  -DPORTURB_LINK_FLAGS="${PORTURB_LINK_FLAGS}"   \
  -DYAKL_ARCH="${YAKL_ARCH}"                     \
  -DYAKL_HAVE_MPI=ON                             \
  -DYAKL_DEBUG="${YAKL_DEBUG}"                   \
  -DYAKL_PROFILE="${YAKL_PROFILE}"               \
  -DYAKL_AUTO_PROFILE="${YAKL_AUTO_PROFILE}"     \
  -DYAKL_VERBOSE="${YAKL_VERBOSE}"               \
  -DYAKL_VERBOSE_FILE="${YAKL_VERBOSE_FILE}"     \
  -DYAKL_AUTO_FENCE="${YAKL_AUTO_FENCE}"         \
  -DCMAKE_CUDA_HOST_COMPILER="${CXX}"            \
  -DMINIWEATHER_ML_HOME="`pwd`/.."               \
  -Wno-dev                                       \
  ..


