# Copyright 2024, AMD. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of AMD nor the names of its contributors may be used
#    to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Common test sources (shared between different test executables)
set(TEST_COMMON_SRCS
  ${CMAKE_SOURCE_DIR}/src/pb_utils.cc
  ${CMAKE_SOURCE_DIR}/src/pb_memory.cc
  ${CMAKE_SOURCE_DIR}/src/pb_exception.h
  ${CMAKE_SOURCE_DIR}/src/shm_manager.cc
  ${CMAKE_SOURCE_DIR}/src/scoped_defer.cc
)

# Extended sources for tensor tests
set(TEST_TENSOR_SRCS
  ${TEST_COMMON_SRCS}
  ${CMAKE_SOURCE_DIR}/src/pb_tensor.cc
  ${CMAKE_SOURCE_DIR}/src/pb_string.cc
  ${CMAKE_SOURCE_DIR}/src/gpu_buffers.cc
)

# Include directories for tests
set(TEST_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}
  ${boostorg_INCLUDE_DIRS}
  ${dlpack_SOURCE_DIR}/include
)

#
# HIPHandler Tests (ROCm-specific tests)
#
if(${TRITON_ENABLE_ROCM})
  add_executable(
    hip_handler_test
    hip_handler_test.cc
    ${TEST_COMMON_SRCS}
  )

  target_include_directories(hip_handler_test PRIVATE ${TEST_INCLUDE_DIRS})

  target_compile_features(hip_handler_test PRIVATE cxx_std_11)
  target_compile_definitions(hip_handler_test PRIVATE TRITON_ENABLE_ROCM)

  target_link_libraries(
    hip_handler_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )

  # hip_handler_test depends on hip_ipc_child for IPC tests
  add_dependencies(hip_handler_test hip_ipc_child)

  add_test(NAME HIPHandlerTest COMMAND hip_handler_test)
  set_tests_properties(HIPHandlerTest PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/rocm/lib:$ENV{LD_LIBRARY_PATH}"
  )
endif()

#
# PbMemory Tests
#
add_executable(
  pb_memory_test
  pb_memory_test.cc
  ${TEST_COMMON_SRCS}
)

target_include_directories(pb_memory_test PRIVATE ${TEST_INCLUDE_DIRS})

target_compile_features(pb_memory_test PRIVATE cxx_std_11)

if(${TRITON_ENABLE_ROCM})
  target_compile_definitions(pb_memory_test PRIVATE TRITON_ENABLE_ROCM)
  target_link_libraries(
    pb_memory_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )
else()
  target_link_libraries(
    pb_memory_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
  )
endif()

add_test(NAME PbMemoryTest COMMAND pb_memory_test)

#
# SharedMemoryManager Tests
#
add_executable(
  shm_manager_test
  shm_manager_test.cc
  ${CMAKE_SOURCE_DIR}/src/shm_manager.cc
  ${CMAKE_SOURCE_DIR}/src/scoped_defer.cc
  ${CMAKE_SOURCE_DIR}/src/pb_memory.cc
  ${CMAKE_SOURCE_DIR}/src/pb_utils.cc
)

target_include_directories(shm_manager_test PRIVATE ${TEST_INCLUDE_DIRS})

target_compile_features(shm_manager_test PRIVATE cxx_std_11)

if(${TRITON_ENABLE_ROCM})
  target_compile_definitions(shm_manager_test PRIVATE TRITON_ENABLE_ROCM)
  target_link_libraries(
    shm_manager_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )
else()
  target_link_libraries(
    shm_manager_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
  )
endif()

add_test(NAME ShmManagerTest COMMAND shm_manager_test)

#
# ScopedSetDevice Tests (ROCm-specific)
#
if(${TRITON_ENABLE_ROCM})
  add_executable(
    scoped_device_test
    scoped_device_test.cc
    ${TEST_COMMON_SRCS}
  )

  target_include_directories(scoped_device_test PRIVATE ${TEST_INCLUDE_DIRS})

  target_compile_features(scoped_device_test PRIVATE cxx_std_11)
  target_compile_definitions(scoped_device_test PRIVATE TRITON_ENABLE_ROCM)

  target_link_libraries(
    scoped_device_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )

  add_test(NAME ScopedDeviceTest COMMAND scoped_device_test)
  set_tests_properties(ScopedDeviceTest PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/rocm/lib:$ENV{LD_LIBRARY_PATH}"
  )
endif()

#
# PbTensor Tests
#
add_executable(
  pb_tensor_test
  pb_tensor_test.cc
  ${TEST_TENSOR_SRCS}
)

target_include_directories(pb_tensor_test PRIVATE ${TEST_INCLUDE_DIRS})

target_compile_features(pb_tensor_test PRIVATE cxx_std_11)

if(${TRITON_ENABLE_ROCM})
  target_compile_definitions(pb_tensor_test PRIVATE TRITON_ENABLE_ROCM)
  target_link_libraries(
    pb_tensor_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      dlpack
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )
else()
  target_link_libraries(
    pb_tensor_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      dlpack
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
  )
endif()

add_test(NAME PbTensorTest COMMAND pb_tensor_test)

#
# GPUBuffers Tests
#
add_executable(
  gpu_buffers_test
  gpu_buffers_test.cc
  ${TEST_TENSOR_SRCS}
)

target_include_directories(gpu_buffers_test PRIVATE ${TEST_INCLUDE_DIRS})

target_compile_features(gpu_buffers_test PRIVATE cxx_std_11)

if(${TRITON_ENABLE_ROCM})
  target_compile_definitions(gpu_buffers_test PRIVATE TRITON_ENABLE_ROCM)
  target_link_libraries(
    gpu_buffers_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      dlpack
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )
else()
  target_link_libraries(
    gpu_buffers_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      dlpack
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
  )
endif()

add_test(NAME GPUBuffersTest COMMAND gpu_buffers_test)

#
# HIP IPC Child Process Executable
# Separate executable used by hip_ipc_test and hip_handler_test for true multi-process IPC testing
#
if(${TRITON_ENABLE_ROCM})
  add_executable(
    hip_ipc_child
    hip_ipc_child.cc
    ${TEST_COMMON_SRCS}
  )

  target_include_directories(hip_ipc_child PRIVATE ${TEST_INCLUDE_DIRS})

  target_compile_features(hip_ipc_child PRIVATE cxx_std_11)
  target_compile_definitions(hip_ipc_child PRIVATE TRITON_ENABLE_ROCM)

  target_link_libraries(
    hip_ipc_child
    PRIVATE
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )
endif()

#
# HIP IPC Multi-Process Test
# Tests IPC handle sharing between processes using exec()
#
if(${TRITON_ENABLE_ROCM})
  add_executable(
    hip_ipc_test
    hip_ipc_test.cc
    ${TEST_COMMON_SRCS}
  )

  target_include_directories(hip_ipc_test PRIVATE ${TEST_INCLUDE_DIRS})

  target_compile_features(hip_ipc_test PRIVATE cxx_std_11)
  target_compile_definitions(hip_ipc_test PRIVATE TRITON_ENABLE_ROCM)

  target_link_libraries(
    hip_ipc_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )

  # hip_ipc_test depends on hip_ipc_child being built
  add_dependencies(hip_ipc_test hip_ipc_child)

  add_test(NAME HipIPCTest COMMAND hip_ipc_test)
  set_tests_properties(HipIPCTest PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/rocm/lib:$ENV{LD_LIBRARY_PATH}"
  )
endif()

#
# ROCm Integration Test (comprehensive HIP API tests)
#
if(${TRITON_ENABLE_ROCM})
  add_executable(
    rocm_integration_test
    rocm_integration_test.cc
    ${TEST_COMMON_SRCS}
  )

  target_include_directories(rocm_integration_test PRIVATE ${TEST_INCLUDE_DIRS})

  target_compile_features(rocm_integration_test PRIVATE cxx_std_11)
  target_compile_definitions(rocm_integration_test PRIVATE TRITON_ENABLE_ROCM)

  target_link_libraries(
    rocm_integration_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )

  add_test(NAME ROCmIntegrationTest COMMAND rocm_integration_test)
  set_tests_properties(ROCmIntegrationTest PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/rocm/lib:$ENV{LD_LIBRARY_PATH}"
  )
endif()

#
# pb_utils Test (utility functions and structures)
#
if(${TRITON_ENABLE_ROCM})
  add_executable(
    pb_utils_test
    pb_utils_test.cc
    ${TEST_COMMON_SRCS}
  )

  # add_executable(
  #   hip_ipc_child
  #   hip_ipc_child.cc
  #   ${TEST_COMMON_SRCS}
  # )

  # target_link_libraries(
  #   hip_ipc_child
  #   PRIVATE
  #     GTest::gtest_main
  #     GTest::gmock
  #     triton-backend-utils
  #     triton-core-serverstub
  #     pb_utils
  #     Threads::Threads
  #     ${CMAKE_DL_LIBS}
  #     -lrt
  #     hip::host
  # )

  target_include_directories(pb_utils_test PRIVATE ${TEST_INCLUDE_DIRS})

  target_compile_features(pb_utils_test PRIVATE cxx_std_11)
  target_compile_definitions(pb_utils_test PRIVATE TRITON_ENABLE_ROCM)

  target_link_libraries(
    pb_utils_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )

  add_test(NAME PbUtilsTest COMMAND pb_utils_test)
  set_tests_properties(PbUtilsTest PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/rocm/lib:$ENV{LD_LIBRARY_PATH}"
  )
endif()

#
# ROCm Backend Patterns Test (patterns from python_be.cc)
#
if(${TRITON_ENABLE_ROCM})
  add_executable(
    rocm_backend_patterns_test
    rocm_backend_patterns_test.cc
    ${TEST_TENSOR_SRCS}
  )

  target_include_directories(rocm_backend_patterns_test PRIVATE ${TEST_INCLUDE_DIRS})

  target_compile_features(rocm_backend_patterns_test PRIVATE cxx_std_11)
  target_compile_definitions(rocm_backend_patterns_test PRIVATE TRITON_ENABLE_ROCM)

  target_link_libraries(
    rocm_backend_patterns_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      dlpack
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )

  add_test(NAME ROCmBackendPatternsTest COMMAND rocm_backend_patterns_test)
  set_tests_properties(ROCmBackendPatternsTest PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/rocm/lib:$ENV{LD_LIBRARY_PATH}"
  )
endif()

#
# Request Executor Test
# Note: Tests utility classes and ROCm patterns without complex dependencies
#
set(REQUEST_EXECUTOR_SRCS
  ${TEST_COMMON_SRCS}
  ${CMAKE_SOURCE_DIR}/src/infer_payload.cc
  ${CMAKE_SOURCE_DIR}/src/pb_tensor.cc
  ${CMAKE_SOURCE_DIR}/src/pb_string.cc
  ${CMAKE_SOURCE_DIR}/src/pb_error.cc
  ${CMAKE_SOURCE_DIR}/src/gpu_buffers.cc
  ${CMAKE_SOURCE_DIR}/src/ipc_message.cc
  ${CMAKE_SOURCE_DIR}/src/pb_utils.cc
)

if(${TRITON_ENABLE_ROCM})
  add_executable(
    request_executor_test
    request_executor_test.cc
    ${REQUEST_EXECUTOR_SRCS}
  )

  target_include_directories(request_executor_test PRIVATE ${TEST_INCLUDE_DIRS})

  target_compile_features(request_executor_test PRIVATE cxx_std_11)
  target_compile_definitions(request_executor_test PRIVATE TRITON_ENABLE_ROCM)

  target_link_libraries(
    request_executor_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      dlpack
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )

  add_test(NAME RequestExecutorTest COMMAND request_executor_test)
  set_tests_properties(RequestExecutorTest PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/rocm/lib:$ENV{LD_LIBRARY_PATH}"
  )
endif()

#
# Python Backend (python_be.h/.cc) Tests
# Tests BackendState struct, utility patterns, and configuration validation
#
add_executable(
  python_be_test
  python_be_test.cc
  ${TEST_COMMON_SRCS}
)

target_include_directories(python_be_test PRIVATE ${TEST_INCLUDE_DIRS})

target_compile_features(python_be_test PRIVATE cxx_std_11)

if(${TRITON_ENABLE_ROCM})
  target_compile_definitions(python_be_test PRIVATE TRITON_ENABLE_ROCM)
  target_link_libraries(
    python_be_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )
else()
  target_link_libraries(
    python_be_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
  )
endif()

add_test(NAME PythonBeTest COMMAND python_be_test)

#
# ScopedDefer Tests (RAII defer pattern)
#
add_executable(
  scoped_defer_test
  scoped_defer_test.cc
  ${CMAKE_SOURCE_DIR}/src/scoped_defer.cc
)

target_include_directories(scoped_defer_test PRIVATE ${TEST_INCLUDE_DIRS})

target_compile_features(scoped_defer_test PRIVATE cxx_std_11)

target_link_libraries(
  scoped_defer_test
  PRIVATE
    GTest::gtest_main
    GTest::gmock
    Threads::Threads
)

add_test(NAME ScopedDeferTest COMMAND scoped_defer_test)

#
# PbString Tests (string storage in shared memory)
#
add_executable(
  pb_string_test
  pb_string_test.cc
  ${CMAKE_SOURCE_DIR}/src/pb_string.cc
  ${CMAKE_SOURCE_DIR}/src/shm_manager.cc
  ${CMAKE_SOURCE_DIR}/src/scoped_defer.cc
  ${CMAKE_SOURCE_DIR}/src/pb_memory.cc
  ${CMAKE_SOURCE_DIR}/src/pb_utils.cc
)

target_include_directories(pb_string_test PRIVATE ${TEST_INCLUDE_DIRS})

target_compile_features(pb_string_test PRIVATE cxx_std_11)

if(${TRITON_ENABLE_ROCM})
  target_compile_definitions(pb_string_test PRIVATE TRITON_ENABLE_ROCM)
  target_link_libraries(
    pb_string_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )
else()
  target_link_libraries(
    pb_string_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
  )
endif()

add_test(NAME PbStringTest COMMAND pb_string_test)

#
# PbError Tests (error handling with shared memory)
#
add_executable(
  pb_error_test
  pb_error_test.cc
  ${CMAKE_SOURCE_DIR}/src/pb_error.cc
  ${CMAKE_SOURCE_DIR}/src/pb_string.cc
  ${CMAKE_SOURCE_DIR}/src/shm_manager.cc
  ${CMAKE_SOURCE_DIR}/src/scoped_defer.cc
  ${CMAKE_SOURCE_DIR}/src/pb_memory.cc
  ${CMAKE_SOURCE_DIR}/src/pb_utils.cc
)

target_include_directories(pb_error_test PRIVATE ${TEST_INCLUDE_DIRS})

target_compile_features(pb_error_test PRIVATE cxx_std_11)

if(${TRITON_ENABLE_ROCM})
  target_compile_definitions(pb_error_test PRIVATE TRITON_ENABLE_ROCM)
  target_link_libraries(
    pb_error_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
      hip::host
  )
else()
  target_link_libraries(
    pb_error_test
    PRIVATE
      GTest::gtest_main
      GTest::gmock
      triton-backend-utils
      triton-core-serverstub
      Threads::Threads
      ${CMAKE_DL_LIBS}
      -lrt
  )
endif()

add_test(NAME PbErrorTest COMMAND pb_error_test)

# Add a custom target to run all tests
add_custom_target(run_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS
    pb_memory_test
    shm_manager_test
    pb_tensor_test
    gpu_buffers_test
    scoped_defer_test
    pb_string_test
    pb_error_test
    $<$<BOOL:${TRITON_ENABLE_ROCM}>:hip_handler_test>
    $<$<BOOL:${TRITON_ENABLE_ROCM}>:scoped_device_test>
    $<$<BOOL:${TRITON_ENABLE_ROCM}>:rocm_integration_test>
    $<$<BOOL:${TRITON_ENABLE_ROCM}>:pb_utils_test>
    $<$<BOOL:${TRITON_ENABLE_ROCM}>:rocm_backend_patterns_test>
    $<$<BOOL:${TRITON_ENABLE_ROCM}>:request_executor_test>
    $<$<BOOL:${TRITON_ENABLE_ROCM}>:hip_ipc_test>
    $<$<BOOL:${TRITON_ENABLE_ROCM}>:hip_ipc_child>
    python_be_test
  COMMENT "Running all unit tests..."
)

