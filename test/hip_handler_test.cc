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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <hip/hip_runtime.h>

#include <cstring>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <linux/limits.h>

#include "pb_utils.h"
#include "pb_exception.h"
#include "triton/core/tritonserver.h"

// Must match the struct in hip_ipc_child.cc
enum class IPCMode {
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
  IPCMode mode;
};

namespace triton { namespace backend { namespace python {
namespace test {

//
// HIPHandler Singleton Tests
//
class HIPHandlerTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Get the singleton instance - this will initialize the HIP runtime
    // if libamdhip64.so is available
    hip_handler_ = &HIPHandler::getInstance();
  }

  HIPHandler* hip_handler_;
};

// Test that HIPHandler singleton is properly initialized
TEST_F(HIPHandlerTest, SingletonInitialization)
{
  // Multiple calls should return the same instance
  HIPHandler& instance1 = HIPHandler::getInstance();
  HIPHandler& instance2 = HIPHandler::getInstance();
  EXPECT_EQ(&instance1, &instance2);
}

// Test IsAvailable returns correct status based on HIP library availability
TEST_F(HIPHandlerTest, IsAvailableReflectsLibraryStatus)
{
  bool available = hip_handler_->IsAvailable();

  if (available) {
    // HIP library was successfully loaded
    EXPECT_TRUE(hip_handler_->GetErrorString().empty());
  }
}

// Test GetErrorString and ClearErrorString
TEST_F(HIPHandlerTest, ErrorStringManagement)
{
  // Initially or after clearing, error string should be empty (if HIP is available)
  hip_handler_->ClearErrorString();
  if (hip_handler_->IsAvailable()) {
    EXPECT_TRUE(hip_handler_->GetErrorString().empty());
  }
}

//
// HIPHandler Memory Operations Tests (requires GPU)
//
class HIPHandlerMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    hip_handler_ = &HIPHandler::getInstance();

    // Skip tests if HIP is not available
    if (!hip_handler_->IsAvailable()) {
      GTEST_SKIP() << "HIP runtime not available, skipping memory tests";
    }

    // Check if we have at least one GPU
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    EXPECT_EQ(err, hipSuccess);
    EXPECT_NE(device_count, 0);

    device_count_ = device_count;
  }

  void TearDown() override
  {
    // Clean up any allocated memory
    for (void* ptr : allocated_ptrs_) {
      (void)hipFree(ptr);
    }
    allocated_ptrs_.clear();
  }

  HIPHandler* hip_handler_;
  int device_count_ = 0;
  std::vector<void*> allocated_ptrs_;
};

// Test MaybeSetDevice with valid device
TEST_F(HIPHandlerMemoryTest, MaybeSetDeviceValidDevice)
{
  // MaybeSetDevice should not throw for device 0 if it has a primary context
  void* dummy_ptr = nullptr;
  hipError_t err = hipMalloc(&dummy_ptr, 1024);
  ASSERT_EQ(err, hipSuccess) << "Failed to allocate memory on device 0";
  allocated_ptrs_.push_back(dummy_ptr);

  EXPECT_NO_THROW(hip_handler_->MaybeSetDevice(0));
}

// Test PointerGetAttribute with valid GPU pointer
TEST_F(HIPHandlerMemoryTest, PointerGetAttributeValidPointer)
{
  // Allocate GPU memory
  void* gpu_ptr = nullptr;
  hipError_t err = hipMalloc(&gpu_ptr, 4096);
  ASSERT_EQ(err, hipSuccess) << "Failed to allocate GPU memory";
  allocated_ptrs_.push_back(gpu_ptr);

  hipDeviceptr_t start_address = 0;
  size_t range_size = 0;
  
  // Get the start address attribute
  EXPECT_NO_THROW(hip_handler_->PointerGetAttribute(
      &start_address, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
      reinterpret_cast<hipDeviceptr_t>(gpu_ptr)));

  // Get the range size attribute
  EXPECT_NO_THROW(hip_handler_->PointerGetAttribute(
      reinterpret_cast<hipDeviceptr_t*>(&range_size),
      HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
      reinterpret_cast<hipDeviceptr_t>(gpu_ptr)));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(start_address),
            reinterpret_cast<uintptr_t>(gpu_ptr));

  // The range size should match the allocation size
  EXPECT_EQ(range_size, static_cast<size_t>(4096));
}

TEST_F(HIPHandlerMemoryTest, IpcHandleGetSucceeds)
{
  // Note: IPC handles can only be *opened* from a different process.
  // Opening in the same process results in "invalid device context".
  // Full IPC open/close testing is done in hip_ipc_test.cc using exec().
  //
  // This test verifies that getting an IPC handle works correctly.

  EXPECT_NO_THROW(hip_handler_->MaybeSetDevice(0));

  // Allocate GPU memory
  void* gpu_ptr = nullptr;
  size_t alloc_size = 4096;
  hipError_t err = hipMalloc(&gpu_ptr, alloc_size);
  ASSERT_EQ(err, hipSuccess) << "Failed to allocate GPU memory";
  allocated_ptrs_.push_back(gpu_ptr);

  // Initialize the memory
  err = hipMemset(gpu_ptr, 0xAB, alloc_size);
  ASSERT_EQ(err, hipSuccess) << "Failed to memset GPU memory";

  err = hipDeviceSynchronize();
  ASSERT_EQ(err, hipSuccess) << "Failed to synchronize";

  // Get IPC handle for that memory - this should succeed
  hipIpcMemHandle_t ipc_handle;
  err = hipIpcGetMemHandle(&ipc_handle, gpu_ptr);
  ASSERT_EQ(err, hipSuccess) << "Failed to get IPC handle: "
                             << hipGetErrorString(err);

  // Verify the handle is non-zero (has been populated)
  // The handle is an opaque struct, but we can check it's not all zeros
  bool handle_is_set = false;
  const unsigned char* handle_bytes =
      reinterpret_cast<const unsigned char*>(&ipc_handle);
  for (size_t i = 0; i < sizeof(ipc_handle); ++i) {
    if (handle_bytes[i] != 0) {
      handle_is_set = true;
      break;
    }
  }
  EXPECT_TRUE(handle_is_set) << "IPC handle appears to be empty";
}

// Test OpenHipHandle and CloseHipHandle using exec-based multi-process testing
// This is the only way to properly test IPC - handles must be opened from a different process
TEST_F(HIPHandlerMemoryTest, IpcOpenCloseViaChildProcess)
{
  EXPECT_NO_THROW(hip_handler_->MaybeSetDevice(0));

  // Find the child executable
  std::string child_exe_path;
  char exe_path[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  if (len != -1) {
    exe_path[len] = '\0';
    std::string path(exe_path);
    size_t last_slash = path.rfind('/');
    if (last_slash != std::string::npos) {
      child_exe_path = path.substr(0, last_slash + 1) + "hip_ipc_child";
    }
  }

  if (child_exe_path.empty() || access(child_exe_path.c_str(), X_OK) != 0) {
    GTEST_SKIP() << "hip_ipc_child executable not found at: " << child_exe_path;
  }

  // Allocate GPU memory and initialize with test data
  const size_t num_elements = 8;
  const size_t byte_size = num_elements * sizeof(float);
  std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  void* gpu_ptr = nullptr;
  hipError_t err = hipMalloc(&gpu_ptr, byte_size);
  ASSERT_EQ(err, hipSuccess) << "Failed to allocate GPU memory";
  allocated_ptrs_.push_back(gpu_ptr);

  err = hipMemcpy(gpu_ptr, test_data.data(), byte_size, hipMemcpyHostToDevice);
  ASSERT_EQ(err, hipSuccess) << "Failed to copy data to GPU";

  err = hipDeviceSynchronize();
  ASSERT_EQ(err, hipSuccess);

  // Get IPC handle
  hipIpcMemHandle_t ipc_handle;
  err = hipIpcGetMemHandle(&ipc_handle, gpu_ptr);
  ASSERT_EQ(err, hipSuccess) << "Failed to get IPC handle";

  // Create params file for child - using HANDLER_VERIFY mode to test HIPHandler
  std::string params_file = "/tmp/hip_handler_test_params_" + std::to_string(getpid());
  IPCChildParams params;
  params.ipc_handle = ipc_handle;
  params.device_id = 0;
  params.num_elements = num_elements;
  memcpy(params.expected_data, test_data.data(), byte_size);
  params.mode = IPCMode::HANDLER_VERIFY;  // Test HIPHandler::OpenHipHandle/CloseHipHandle

  std::ofstream out(params_file, std::ios::binary);
  ASSERT_TRUE(out.good()) << "Failed to create params file";
  out.write(reinterpret_cast<const char*>(&params), sizeof(params));
  out.close();

  // Fork and exec child process
  pid_t pid = fork();
  ASSERT_NE(pid, -1) << "Fork failed";

  if (pid == 0) {
    // Child: exec the child executable
    execl(child_exe_path.c_str(), "hip_ipc_child", params_file.c_str(), nullptr);
    _exit(99);  // If exec fails
  }

  // Parent: wait for child
  int status;
  waitpid(pid, &status, 0);

  // Clean up params file
  unlink(params_file.c_str());

  ASSERT_TRUE(WIFEXITED(status)) << "Child did not exit normally";
  int exit_code = WEXITSTATUS(status);
  ASSERT_EQ(exit_code, 0) << "Child process failed with exit code: " << exit_code
                          << "\nExit codes: 4=MaybeSetDevice, 5=OpenHipHandle, "
                          << "6=verification/memcpy, 7=CloseHipHandle, 8=HIPHandler unavailable";
}

// Test OpenHipHandle and CloseHipHandle with write-back to verify bidirectional IPC
TEST_F(HIPHandlerMemoryTest, IpcOpenCloseWriteBackViaChildProcess)
{
  EXPECT_NO_THROW(hip_handler_->MaybeSetDevice(0));

  // Find the child executable
  std::string child_exe_path;
  char exe_path[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  if (len != -1) {
    exe_path[len] = '\0';
    std::string path(exe_path);
    size_t last_slash = path.rfind('/');
    if (last_slash != std::string::npos) {
      child_exe_path = path.substr(0, last_slash + 1) + "hip_ipc_child";
    }
  }

  if (child_exe_path.empty() || access(child_exe_path.c_str(), X_OK) != 0) {
    GTEST_SKIP() << "hip_ipc_child executable not found at: " << child_exe_path;
  }

  // Allocate GPU memory and initialize
  const size_t num_elements = 4;
  const size_t byte_size = num_elements * sizeof(float);
  std::vector<float> test_data = {10.0f, 20.0f, 30.0f, 40.0f};

  void* gpu_ptr = nullptr;
  hipError_t err = hipMalloc(&gpu_ptr, byte_size);
  ASSERT_EQ(err, hipSuccess);
  allocated_ptrs_.push_back(gpu_ptr);

  err = hipMemcpy(gpu_ptr, test_data.data(), byte_size, hipMemcpyHostToDevice);
  ASSERT_EQ(err, hipSuccess);

  err = hipDeviceSynchronize();
  ASSERT_EQ(err, hipSuccess);

  hipIpcMemHandle_t ipc_handle;
  err = hipIpcGetMemHandle(&ipc_handle, gpu_ptr);
  ASSERT_EQ(err, hipSuccess);

  // Use HANDLER_WRITE_BACK mode - child will multiply data by 2 using HIPHandler
  std::string params_file = "/tmp/hip_handler_test_params_wb_" + std::to_string(getpid());
  IPCChildParams params;
  params.ipc_handle = ipc_handle;
  params.device_id = 0;
  params.num_elements = num_elements;
  params.mode = IPCMode::HANDLER_WRITE_BACK;

  std::ofstream out(params_file, std::ios::binary);
  ASSERT_TRUE(out.good());
  out.write(reinterpret_cast<const char*>(&params), sizeof(params));
  out.close();

  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    execl(child_exe_path.c_str(), "hip_ipc_child", params_file.c_str(), nullptr);
    _exit(99);
  }

  int status;
  waitpid(pid, &status, 0);
  unlink(params_file.c_str());

  ASSERT_TRUE(WIFEXITED(status));
  int exit_code = WEXITSTATUS(status);
  ASSERT_EQ(exit_code, 0) << "Child failed with exit code: " << exit_code;

  // Verify parent can read the modified data (child multiplied by 2)
  std::vector<float> result_data(num_elements);
  err = hipMemcpy(result_data.data(), gpu_ptr, byte_size, hipMemcpyDeviceToHost);
  ASSERT_EQ(err, hipSuccess);

  for (size_t i = 0; i < num_elements; ++i) {
    EXPECT_FLOAT_EQ(result_data[i], test_data[i] * 2.0f)
        << "Data mismatch at index " << i;
  }
}



//
// Multi-Device Tests
//
class HIPHandlerMultiDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    hip_handler_ = &HIPHandler::getInstance();

    if (!hip_handler_->IsAvailable()) {
      GTEST_SKIP() << "HIP runtime not available";
    }

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count < 2) {
      GTEST_SKIP() << "Multi-device tests require at least 2 GPUs";
    }

    device_count_ = device_count;
  }

  HIPHandler* hip_handler_;
  int device_count_ = 0;
};

// Test MaybeSetDevice across multiple devices
TEST_F(HIPHandlerMultiDeviceTest, MaybeSetDeviceMultipleDevices)
{
  // Create contexts on both devices
  for (int i = 0; i < device_count_; ++i) {
    ASSERT_EQ(hipSetDevice(i), hipSuccess);
    void* ptr = nullptr;
    ASSERT_EQ(hipMalloc(&ptr, 1024), hipSuccess);
    (void)hipFree(ptr);
  }

  // Now test MaybeSetDevice on each device
  for (int i = 0; i < device_count_; ++i) {
    EXPECT_NO_THROW(hip_handler_->MaybeSetDevice(i));

    int current_device = -1;
    ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
    EXPECT_EQ(current_device, i);
  }
}

//
// Error Handling Tests
//
class HIPHandlerErrorTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    hip_handler_ = &HIPHandler::getInstance();
    if (!hip_handler_->IsAvailable()) {
      GTEST_SKIP() << "HIP runtime not available";
    }
  }

  HIPHandler* hip_handler_;
};

// Test PointerGetAttribute with invalid pointer throws exception
TEST_F(HIPHandlerErrorTest, PointerGetAttributeInvalidPointer)
{
  hipDeviceptr_t start_address = 0;
  // Using a non-GPU pointer should throw
  void* invalid_ptr = reinterpret_cast<void*>(0xDEADBEEF);

  EXPECT_THROW(
      hip_handler_->PointerGetAttribute(
          &start_address, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
          reinterpret_cast<hipDeviceptr_t>(invalid_ptr)),
      PythonBackendException);
}

// Test CloseHipHandle with invalid pointer throws exception
TEST_F(HIPHandlerErrorTest, CloseHipHandleInvalidPointer)
{
  void* invalid_ptr = reinterpret_cast<void*>(0xDEADBEEF);

  EXPECT_THROW(
      hip_handler_->CloseHipHandle(0, invalid_ptr),
      PythonBackendException);
}

}  // namespace test
}}}  // namespace triton::backend::python

