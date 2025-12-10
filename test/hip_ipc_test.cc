// Copyright 2024, AMD. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of AMD nor the names of its contributors may be used
//    to endorse or promote products derived from this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Multi-process HIP IPC Tests
// Tests HIP IPC handle creation, sharing, and data verification across processes.
// Uses exec() to spawn a separate child process (hip_ipc_child) for proper IPC testing.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <hip/hip_runtime.h>

#include "pb_exception.h"
#include "pb_utils.h"

namespace triton { namespace backend { namespace python {
namespace test {

// Must match the struct in hip_ipc_child.cc
enum class Mode {
  VERIFY,              // Open handle, verify data, close (raw HIP API)
  WRITE_BACK,          // Open handle, modify data, close (raw HIP API)
  HANDLER_VERIFY,      // Same as VERIFY but using HIPHandler
  HANDLER_WRITE_BACK,  // Same as WRITE_BACK but using HIPHandler
};

struct IPCChildParams {
  hipIpcMemHandle_t ipc_handle;
  int device_id;
  size_t num_elements;
  float expected_data[16];
  Mode mode;
};

class HipIPCTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Check for HIP device availability
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }

    err = hipSetDevice(0);
    if (err != hipSuccess) {
      GTEST_SKIP() << "Failed to set HIP device";
    }

    // Find the child executable
    // It should be in the same directory as this test
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len != -1) {
      exe_path[len] = '\0';
      std::string path(exe_path);
      size_t last_slash = path.rfind('/');
      if (last_slash != std::string::npos) {
        child_exe_path_ = path.substr(0, last_slash + 1) + "hip_ipc_child";
      }
    }

    if (child_exe_path_.empty() || access(child_exe_path_.c_str(), X_OK) != 0) {
      GTEST_SKIP() << "hip_ipc_child executable not found at: " << child_exe_path_;
    }

    // Create temp file for passing params
    params_file_ = "/tmp/hip_ipc_test_params_" + std::to_string(getpid());
  }

  void TearDown() override
  {
    // Clean up temp file
    if (!params_file_.empty()) {
      unlink(params_file_.c_str());
    }

    // Clean up any GPU allocations
    for (void* ptr : gpu_allocations_) {
      (void)hipFree(ptr);
    }
    gpu_allocations_.clear();
  }

  void* AllocateGPU(size_t size)
  {
    void* ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, size);
    if (err == hipSuccess && ptr != nullptr) {
      gpu_allocations_.push_back(ptr);
    }
    return ptr;
  }

  // Run child process and return exit code
  int RunChildProcess()
  {
    pid_t pid = fork();
    if (pid == -1) {
      return -1;  // Fork failed
    }

    if (pid == 0) {
      // Child: exec the child executable
      execl(child_exe_path_.c_str(), "hip_ipc_child", params_file_.c_str(), nullptr);
      // If exec fails, exit with error
      _exit(99);
    }

    // Parent: wait for child
    int status;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status)) {
      return WEXITSTATUS(status);
    }
    return -1;  // Child didn't exit normally
  }

  // Write params to file for child
  bool WriteParams(const IPCChildParams& params)
  {
    std::ofstream out(params_file_, std::ios::binary);
    if (!out) {
      return false;
    }
    out.write(reinterpret_cast<const char*>(&params), sizeof(params));
    return out.good();
  }

  std::string child_exe_path_;
  std::string params_file_;
  std::vector<void*> gpu_allocations_;
};

TEST_F(HipIPCTest, BasicIPCHandleSharing)
{
  // Allocate GPU memory and initialize with test data
  const size_t num_elements = 8;
  const size_t byte_size = num_elements * sizeof(float);
  std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  void* gpu_ptr = AllocateGPU(byte_size);
  ASSERT_NE(gpu_ptr, nullptr) << "Failed to allocate GPU memory";

  hipError_t err = hipMemcpy(
      gpu_ptr, test_data.data(), byte_size, hipMemcpyHostToDevice);
  ASSERT_EQ(err, hipSuccess) << "Failed to copy data to GPU";

  // Ensure data is committed
  err = hipDeviceSynchronize();
  ASSERT_EQ(err, hipSuccess) << "Failed to synchronize";

  // Get IPC handle
  hipIpcMemHandle_t ipc_handle;
  err = hipIpcGetMemHandle(&ipc_handle, gpu_ptr);
  ASSERT_EQ(err, hipSuccess) << "Failed to get IPC handle: "
                             << hipGetErrorString(err);

  // Prepare params for child
  IPCChildParams params;
  params.ipc_handle = ipc_handle;
  params.device_id = 0;
  params.num_elements = num_elements;
  memcpy(params.expected_data, test_data.data(), byte_size);
  params.mode = Mode::VERIFY;

  ASSERT_TRUE(WriteParams(params)) << "Failed to write params file";

  // Run child process
  int exit_code = RunChildProcess();

  ASSERT_EQ(exit_code, 0) << "Child process failed with exit code: " << exit_code
                          << "\nExit codes: 4=hipSetDevice, 5=hipIpcOpenMemHandle, "
                          << "6=verification/memcpy, 7=hipIpcCloseMemHandle";
}

TEST_F(HipIPCTest, IPCWriteBackVerification)
{
  // Test that child can write to shared GPU memory and parent can read it back
  const size_t num_elements = 4;
  const size_t byte_size = num_elements * sizeof(float);
  std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f};

  void* gpu_ptr = AllocateGPU(byte_size);
  ASSERT_NE(gpu_ptr, nullptr);

  hipError_t err = hipMemcpy(
      gpu_ptr, test_data.data(), byte_size, hipMemcpyHostToDevice);
  ASSERT_EQ(err, hipSuccess);

  err = hipDeviceSynchronize();
  ASSERT_EQ(err, hipSuccess);

  hipIpcMemHandle_t ipc_handle;
  err = hipIpcGetMemHandle(&ipc_handle, gpu_ptr);
  ASSERT_EQ(err, hipSuccess) << "Failed to get IPC handle: "
                             << hipGetErrorString(err);

  // Prepare params for child (write-back mode)
  IPCChildParams params;
  params.ipc_handle = ipc_handle;
  params.device_id = 0;
  params.num_elements = num_elements;
  params.mode = Mode::WRITE_BACK;

  ASSERT_TRUE(WriteParams(params)) << "Failed to write params file";

  // Run child process
  int exit_code = RunChildProcess();

  ASSERT_EQ(exit_code, 0) << "Child process failed with exit code: " << exit_code;

  // Verify parent can read the modified data (child multiplied by 2)
  std::vector<float> result_data(num_elements);
  err = hipMemcpy(
      result_data.data(), gpu_ptr, byte_size, hipMemcpyDeviceToHost);
  ASSERT_EQ(err, hipSuccess);

  for (size_t i = 0; i < num_elements; ++i) {
    EXPECT_FLOAT_EQ(result_data[i], test_data[i] * 2.0f)
        << "Data mismatch at index " << i;
  }
}

TEST_F(HipIPCTest, LargerDataTransfer)
{
  // Test with larger data to ensure IPC works for realistic tensor sizes
  const size_t num_elements = 1024;
  const size_t byte_size = num_elements * sizeof(float);

  std::vector<float> test_data(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    test_data[i] = static_cast<float>(i) * 0.5f;
  }

  void* gpu_ptr = AllocateGPU(byte_size);
  ASSERT_NE(gpu_ptr, nullptr);

  hipError_t err = hipMemcpy(
      gpu_ptr, test_data.data(), byte_size, hipMemcpyHostToDevice);
  ASSERT_EQ(err, hipSuccess);

  err = hipDeviceSynchronize();
  ASSERT_EQ(err, hipSuccess);

  hipIpcMemHandle_t ipc_handle;
  err = hipIpcGetMemHandle(&ipc_handle, gpu_ptr);
  ASSERT_EQ(err, hipSuccess) << "Failed to get IPC handle: "
                             << hipGetErrorString(err);

  // Child can only verify first 16 elements (buffer size limit)
  IPCChildParams params;
  params.ipc_handle = ipc_handle;
  params.device_id = 0;
  params.num_elements = 16;
  memcpy(params.expected_data, test_data.data(), 16 * sizeof(float));
  params.mode = Mode::VERIFY;

  ASSERT_TRUE(WriteParams(params)) << "Failed to write params file";

  int exit_code = RunChildProcess();

  ASSERT_EQ(exit_code, 0) << "Child process failed with exit code: " << exit_code;
}

TEST_F(HipIPCTest, HIPHandlerIPCIntegration)
{
  // Test using the HIPHandler class for getting IPC handles
  HIPHandler& handler = HIPHandler::getInstance();

  if (!handler.IsAvailable()) {
    GTEST_SKIP() << "HIPHandler not available";
  }

  const size_t num_elements = 8;
  const size_t byte_size = num_elements * sizeof(float);
  std::vector<float> test_data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};

  void* gpu_ptr = AllocateGPU(byte_size);
  ASSERT_NE(gpu_ptr, nullptr);

  hipError_t err = hipMemcpy(
      gpu_ptr, test_data.data(), byte_size, hipMemcpyHostToDevice);
  ASSERT_EQ(err, hipSuccess);

  err = hipDeviceSynchronize();
  ASSERT_EQ(err, hipSuccess);

  // Get IPC handle using raw HIP API (HIPHandler doesn't have GetHipHandle)
  hipIpcMemHandle_t ipc_handle;
  err = hipIpcGetMemHandle(&ipc_handle, gpu_ptr);
  ASSERT_EQ(err, hipSuccess) << "Failed to get IPC handle";

  IPCChildParams params;
  params.ipc_handle = ipc_handle;
  params.device_id = 0;
  params.num_elements = num_elements;
  memcpy(params.expected_data, test_data.data(), byte_size);
  params.mode = Mode::VERIFY;

  ASSERT_TRUE(WriteParams(params)) << "Failed to write params file";

  int exit_code = RunChildProcess();

  ASSERT_EQ(exit_code, 0) << "Child process failed with exit code: " << exit_code;
}

TEST_F(HipIPCTest, MultipleIPCOperations)
{
  // Test multiple IPC operations in sequence
  const size_t num_elements = 4;
  const size_t byte_size = num_elements * sizeof(float);

  for (int iteration = 0; iteration < 3; ++iteration) {
    std::vector<float> test_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      test_data[i] = static_cast<float>(iteration * 10 + i);
    }

    void* gpu_ptr = nullptr;
    hipError_t err = hipMalloc(&gpu_ptr, byte_size);
    ASSERT_EQ(err, hipSuccess);

    err = hipMemcpy(gpu_ptr, test_data.data(), byte_size, hipMemcpyHostToDevice);
    ASSERT_EQ(err, hipSuccess);

    err = hipDeviceSynchronize();
    ASSERT_EQ(err, hipSuccess);

    hipIpcMemHandle_t ipc_handle;
    err = hipIpcGetMemHandle(&ipc_handle, gpu_ptr);
    ASSERT_EQ(err, hipSuccess) << "Iteration " << iteration
                               << ": Failed to get IPC handle";

    IPCChildParams params;
    params.ipc_handle = ipc_handle;
    params.device_id = 0;
    params.num_elements = num_elements;
    memcpy(params.expected_data, test_data.data(), byte_size);
    params.mode = Mode::VERIFY;

    ASSERT_TRUE(WriteParams(params)) << "Iteration " << iteration
                                     << ": Failed to write params";

    int exit_code = RunChildProcess();
    ASSERT_EQ(exit_code, 0) << "Iteration " << iteration
                            << ": Child failed with exit code " << exit_code;

    (void)hipFree(gpu_ptr);
  }
}

}  // namespace test
}}}  // namespace triton::backend::python
