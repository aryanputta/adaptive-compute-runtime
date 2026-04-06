cmake_minimum_required(VERSION 3.18)
project(adaptive_compute_runtime LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type default ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Options ───────────────────────────────────────────────────────────────────
option(ACR_ENABLE_CUDA    "Build CUDA GPU executor"      ON)
option(ACR_ENABLE_OPENMP  "Enable OpenMP CPU parallel"   ON)
option(ACR_BUILD_TESTS    "Build unit and integration tests" ON)
option(ACR_BUILD_BENCH    "Build benchmark driver"       ON)

# ── CUDA ──────────────────────────────────────────────────────────────────────
if(ACR_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)  # Turing through Hopper
    set(HAVE_CUDA TRUE)
    message(STATUS "CUDA enabled — architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(STATUS "CUDA compiler not found — building CPU-only")
    set(HAVE_CUDA FALSE)
  endif()
else()
  set(HAVE_CUDA FALSE)
endif()

# ── OpenMP ────────────────────────────────────────────────────────────────────
if(ACR_ENABLE_OPENMP)
  find_package(OpenMP)
endif()

# ── Compile flags ─────────────────────────────────────────────────────────────
set(ACR_CXX_FLAGS
    -Wall -Wextra -Wpedantic
    -march=native
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g -O0>
)

# ── Library sources ───────────────────────────────────────────────────────────
set(ACR_SOURCES
    src/runtime/runtime.cpp
    src/runtime/feature_extractor.cpp
    src/scheduler/scheduler.cpp
    src/cpu/cpu_executor.cpp
    src/memory/memory_manager.cpp
    src/profiling/profiler.cpp
    src/gpu/gpu_executor.cpp
)

if(HAVE_CUDA)
  list(APPEND ACR_SOURCES src/gpu/gpu_kernels.cu)
endif()

add_library(acr_lib STATIC ${ACR_SOURCES})

target_include_directories(acr_lib PUBLIC include)

target_compile_options(acr_lib PRIVATE ${ACR_CXX_FLAGS})

if(HAVE_CUDA)
  target_compile_definitions(acr_lib PUBLIC HAVE_CUDA)
  target_link_libraries(acr_lib PUBLIC CUDA::cudart)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(acr_lib PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(acr_lib PUBLIC _OPENMP)
endif()

# ── Benchmark driver ──────────────────────────────────────────────────────────
if(ACR_BUILD_BENCH)
  add_executable(acr_bench benchmarks/bench_main.cpp)
  target_link_libraries(acr_bench PRIVATE acr_lib)
  target_compile_options(acr_bench PRIVATE ${ACR_CXX_FLAGS})
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(ACR_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  set(ACR_TEST_SOURCES
      tests/unit/test_scheduler.cpp
      tests/unit/test_feature_extractor.cpp
      tests/unit/test_memory_manager.cpp
      tests/integration/test_runtime_e2e.cpp
      tests/performance/test_perf_regression.cpp
  )

  add_executable(acr_tests ${ACR_TEST_SOURCES})
  target_link_libraries(acr_tests PRIVATE acr_lib GTest::gtest_main)
  target_compile_options(acr_tests PRIVATE ${ACR_CXX_FLAGS})

  include(GoogleTest)
  gtest_discover_tests(acr_tests)
endif()
