cmake_minimum_required(VERSION 3.22)

project(exafmm-ryoanji CXX)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CSTONE_DIR ${PROJECT_SOURCE_DIR}/cstone/include)
set(CSTONE_TEST_DIR ${PROJECT_SOURCE_DIR}/cstone/test)

add_subdirectory(cstone)
add_subdirectory(ryoanji)
