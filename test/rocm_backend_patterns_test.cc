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
// Tests for ROCm patterns used in python_be.cc
// These tests validate the GPU operations and stream handling patterns
// that are critical for the Python backend's ROCm support.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#ifdef TRITON_ENABLE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

#include "pb_utils.h"
#include "pb_memory.h"
#include "pb_exception.h"
#include "shm_manager.h"
#include "gpu_buffers.h"

namespace triton { namespace backend { namespace python {
namespace test {

#ifdef TRITON_ENABLE_ROCM

//
// Test stream-based memory operations (pattern from python_be.cc)
//
class StreamMemoryOperationsTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }
    device_count_ = device_count;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream_), hipSuccess);
  }

  void TearDown() override
  {
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
    }
    for (void* ptr : allocations_) {
      (void)hipFree(ptr);
    }
    allocations_.clear();
  }

  void* AllocateGPU(size_t size)
  {
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size) == hipSuccess) {
      allocations_.push_back(ptr);
    }
    return ptr;
  }

  hipStream_t stream_ = nullptr;
  int device_count_ = 0;
  std::vector<void*> allocations_;
};

// Test async H2D copy followed by sync (pattern from GetInputTensor)
TEST_F(StreamMemoryOperationsTest, AsyncH2DWithStreamSync)
{
  size_t size = 4096;
  std::vector<float> host_data(size / sizeof(float), 1.234f);

  void* gpu_ptr = AllocateGPU(size);
  ASSERT_NE(gpu_ptr, nullptr);

  // Async copy pattern from python_be.cc
  ASSERT_EQ(hipMemcpyAsync(gpu_ptr, host_data.data(), size,
                           hipMemcpyHostToDevice, stream_), hipSuccess);

  // Stream sync pattern from python_be.cc
  ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);

  // Verify data
  std::vector<float> verify(size / sizeof(float));
  ASSERT_EQ(hipMemcpy(verify.data(), gpu_ptr, size,
                      hipMemcpyDeviceToHost), hipSuccess);

  for (size_t i = 0; i < verify.size(); ++i) {
    EXPECT_FLOAT_EQ(verify[i], 1.234f);
  }
}

// Test async D2H copy followed by sync (pattern from ResponseSendDecoupled)
TEST_F(StreamMemoryOperationsTest, AsyncD2HWithStreamSync)
{
  size_t size = 4096;
  void* gpu_ptr = AllocateGPU(size);
  ASSERT_NE(gpu_ptr, nullptr);

  // Initialize GPU memory
  std::vector<int> init_data(size / sizeof(int), 42);
  ASSERT_EQ(hipMemcpy(gpu_ptr, init_data.data(), size,
                      hipMemcpyHostToDevice), hipSuccess);

  // Async D2H pattern
  std::vector<int> host_data(size / sizeof(int));
  ASSERT_EQ(hipMemcpyAsync(host_data.data(), gpu_ptr, size,
                           hipMemcpyDeviceToHost, stream_), hipSuccess);

  // Stream sync
  ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);

  for (size_t i = 0; i < host_data.size(); ++i) {
    EXPECT_EQ(host_data[i], 42);
  }
}

// Test pattern: multiple async ops then single sync
TEST_F(StreamMemoryOperationsTest, MultipleAsyncOpsSingleSync)
{
  const int num_buffers = 4;
  size_t size = 1024;

  std::vector<void*> gpu_ptrs;
  std::vector<std::vector<char>> host_data(num_buffers);

  for (int i = 0; i < num_buffers; ++i) {
    void* ptr = AllocateGPU(size);
    ASSERT_NE(ptr, nullptr);
    gpu_ptrs.push_back(ptr);

    host_data[i].resize(size, static_cast<char>(i + 1));
    ASSERT_EQ(hipMemcpyAsync(ptr, host_data[i].data(), size,
                             hipMemcpyHostToDevice, stream_), hipSuccess);
  }

  // Single sync for all operations
  ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);

  // Verify all buffers
  for (int i = 0; i < num_buffers; ++i) {
    std::vector<char> verify(size);
    ASSERT_EQ(hipMemcpy(verify.data(), gpu_ptrs[i], size,
                        hipMemcpyDeviceToHost), hipSuccess);
    for (size_t j = 0; j < size; ++j) {
      EXPECT_EQ(verify[j], static_cast<char>(i + 1));
    }
  }
}

//
// Test GPU allocation patterns (from python_be.cc GetInputTensor)
//
class GPUAllocationPatternsTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
  }

  void TearDown() override
  {
    for (void* ptr : allocations_) {
      (void)hipFree(ptr);
    }
    allocations_.clear();
  }

  std::vector<void*> allocations_;
};

// Test hipMalloc pattern from GetInputTensor
TEST_F(GPUAllocationPatternsTest, AllocateForInputTensor)
{
  size_t input_byte_size = 1024 * 1024;  // 1MB typical tensor
  void* dev_ptr = nullptr;

  // Pattern from python_be.cc
  hipError_t err = hipMalloc(&dev_ptr, input_byte_size);
  EXPECT_EQ(err, hipSuccess);
  EXPECT_NE(dev_ptr, nullptr);

  if (dev_ptr != nullptr) {
    allocations_.push_back(dev_ptr);
  }
}

// Test allocation and immediate use pattern
TEST_F(GPUAllocationPatternsTest, AllocateAndUse)
{
  size_t size = 4096;
  void* dev_ptr = nullptr;

  ASSERT_EQ(hipMalloc(&dev_ptr, size), hipSuccess);
  allocations_.push_back(dev_ptr);

  // Immediate memset (common initialization pattern)
  ASSERT_EQ(hipMemset(dev_ptr, 0, size), hipSuccess);

  // Verify initialization
  std::vector<char> host_data(size);
  ASSERT_EQ(hipMemcpy(host_data.data(), dev_ptr, size,
                      hipMemcpyDeviceToHost), hipSuccess);

  for (char c : host_data) {
    EXPECT_EQ(c, 0);
  }
}

// Test multiple small allocations (batch processing pattern)
TEST_F(GPUAllocationPatternsTest, BatchAllocations)
{
  const int batch_size = 8;
  const size_t tensor_size = 512;

  for (int i = 0; i < batch_size; ++i) {
    void* ptr = nullptr;
    ASSERT_EQ(hipMalloc(&ptr, tensor_size), hipSuccess);
    ASSERT_NE(ptr, nullptr);
    allocations_.push_back(ptr);
  }

  EXPECT_EQ(allocations_.size(), batch_size);
}

//
// Test IPC memory handle patterns (used in python_be.cc for tensor transfer)
//
class IPCMemoryPatternsTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    hip_handler_ = &HIPHandler::getInstance();
    if (!hip_handler_->IsAvailable()) {
      GTEST_SKIP() << "HIP handler not available";
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
  }

  void TearDown() override
  {
    for (void* ptr : allocations_) {
      (void)hipFree(ptr);
    }
    allocations_.clear();
  }

  HIPHandler* hip_handler_;
  std::vector<void*> allocations_;
};

// Test getting IPC handle for GPU tensor
TEST_F(IPCMemoryPatternsTest, GetIpcHandleForTensor)
{
  size_t size = 4096;
  void* gpu_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&gpu_ptr, size), hipSuccess);
  allocations_.push_back(gpu_ptr);

  hipIpcMemHandle_t handle;
  hipError_t err = hipIpcGetMemHandle(&handle, gpu_ptr);

  if (err == hipErrorNotSupported) {
    GTEST_SKIP() << "IPC not supported";
  }

  EXPECT_EQ(err, hipSuccess);
}

// Test pointer attribute retrieval (used for offset calculation)
TEST_F(IPCMemoryPatternsTest, GetPointerBaseAddress)
{
  size_t size = 8192;
  void* gpu_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&gpu_ptr, size), hipSuccess);
  allocations_.push_back(gpu_ptr);

  // Get pointer at offset
  void* offset_ptr = static_cast<char*>(gpu_ptr) + 1024;

  hipDeviceptr_t start_address = 0;
  EXPECT_NO_THROW(hip_handler_->PointerGetAttribute(
      &start_address, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
      reinterpret_cast<hipDeviceptr_t>(offset_ptr)));

  // Start address should be <= allocated pointer (might be same or before due to alignment)
  EXPECT_LE(reinterpret_cast<uintptr_t>(start_address),
            reinterpret_cast<uintptr_t>(gpu_ptr));
}

//
// Test device context patterns (from python_be.cc)
//
class DeviceContextPatternsTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }
    device_count_ = device_count;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
  }

  void TearDown() override
  {
    for (void* ptr : allocations_) {
      (void)hipFree(ptr);
    }
    (void)hipSetDevice(0);
  }

  int device_count_ = 0;
  std::vector<void*> allocations_;
};

// Test ScopedSetDevice with memory operations
TEST_F(DeviceContextPatternsTest, ScopedDeviceWithMemOps)
{
  void* ptr = nullptr;
  {
    ScopedSetDevice scoped(0);
    ASSERT_EQ(hipMalloc(&ptr, 1024), hipSuccess);
    allocations_.push_back(ptr);
  }
  EXPECT_NE(ptr, nullptr);
}

// Test device switching for multi-GPU scenarios
TEST_F(DeviceContextPatternsTest, MultiDeviceMemoryAccess)
{
  if (device_count_ < 2) {
    GTEST_SKIP() << "Multi-GPU test requires at least 2 GPUs";
  }

  // Allocate on device 0
  void* ptr0 = nullptr;
  {
    ScopedSetDevice scoped(0);
    ASSERT_EQ(hipMalloc(&ptr0, 1024), hipSuccess);
    allocations_.push_back(ptr0);

    int current;
    ASSERT_EQ(hipGetDevice(&current), hipSuccess);
    EXPECT_EQ(current, 0);
  }

  // Allocate on device 1
  void* ptr1 = nullptr;
  {
    ScopedSetDevice scoped(1);
    ASSERT_EQ(hipMalloc(&ptr1, 1024), hipSuccess);
    allocations_.push_back(ptr1);

    int current;
    ASSERT_EQ(hipGetDevice(&current), hipSuccess);
    EXPECT_EQ(current, 1);
  }

  // Back to device 0
  int final_device;
  ASSERT_EQ(hipGetDevice(&final_device), hipSuccess);
  EXPECT_EQ(final_device, 0);

  EXPECT_NE(ptr0, ptr1);
}

//
// Test GPU buffer patterns used in python_be.cc BLS execution
//
class GPUBuffersPatternsTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    shm_region_name_ = "/gpu_buffers_patterns_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_, 4 * 1024 * 1024, 1024 * 1024, true);
  }

  void TearDown() override
  {
    for (void* ptr : allocations_) {
      (void)hipFree(ptr);
    }
    allocations_.clear();
    shm_pool_.reset();
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
  std::vector<void*> allocations_;
};

// Test GPUBuffersHelper pattern used in ExecuteBLSRequest
TEST_F(GPUBuffersPatternsTest, GPUBuffersHelperWorkflow)
{
  GPUBuffersHelper gpu_buffer_helper;

  // Simulate adding GPU buffers for BLS request
  void* gpu_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&gpu_ptr, 1024), hipSuccess);
  allocations_.push_back(gpu_ptr);

  // Create PbMemory and add its handle
  auto memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_GPU,
      static_cast<int64_t>(0),
      static_cast<uint64_t>(1024),
      reinterpret_cast<char*>(gpu_ptr));

  gpu_buffer_helper.AddBuffer(memory->ShmHandle());

  // Complete the helper
  gpu_buffer_helper.Complete(shm_pool_);

  EXPECT_NE(gpu_buffer_helper.ShmHandle(), 0);
}

// Test error setting in GPUBuffersHelper
TEST_F(GPUBuffersPatternsTest, GPUBuffersHelperWithError)
{
  GPUBuffersHelper gpu_buffer_helper;

  // Set an error
  gpu_buffer_helper.SetError(shm_pool_, "Test error message");

  // Complete should still work
  gpu_buffer_helper.Complete(shm_pool_);

  EXPECT_NE(gpu_buffer_helper.ShmHandle(), 0);
}

//
// Test stream synchronization patterns for output processing
//
class OutputProcessingPatternsTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream_), hipSuccess);
  }

  void TearDown() override
  {
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
    }
    for (void* ptr : allocations_) {
      (void)hipFree(ptr);
    }
    allocations_.clear();
  }

  void* AllocateGPU(size_t size)
  {
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size) == hipSuccess) {
      allocations_.push_back(ptr);
    }
    return ptr;
  }

  hipStream_t stream_ = nullptr;
  std::vector<void*> allocations_;
};

// Test output buffer copy pattern (from ProcessRequests)
TEST_F(OutputProcessingPatternsTest, OutputBufferCopyPattern)
{
  size_t size = 4096;

  // Simulate output tensor in GPU
  void* gpu_output = AllocateGPU(size);
  ASSERT_NE(gpu_output, nullptr);

  // Initialize with some data
  std::vector<float> init_data(size / sizeof(float), 9.99f);
  ASSERT_EQ(hipMemcpy(gpu_output, init_data.data(), size,
                      hipMemcpyHostToDevice), hipSuccess);

  // Simulate copy to output buffer (pattern from ProcessRequests)
  std::vector<float> output_buffer(size / sizeof(float));
  bool cuda_copy = false;

  ASSERT_EQ(hipMemcpyAsync(output_buffer.data(), gpu_output, size,
                           hipMemcpyDeviceToHost, stream_), hipSuccess);
  cuda_copy = true;

  if (cuda_copy) {
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
  }

  for (const auto& val : output_buffer) {
    EXPECT_FLOAT_EQ(val, 9.99f);
  }
}

// Test conditional stream sync (pattern from ResponseSendDecoupled)
TEST_F(OutputProcessingPatternsTest, ConditionalStreamSync)
{
  bool cuda_copy = false;

  // First case: no GPU operations
  if (cuda_copy) {
    FAIL() << "Should not sync when cuda_copy is false";
  }

  // Second case: with GPU operations
  void* gpu_ptr = AllocateGPU(1024);
  ASSERT_NE(gpu_ptr, nullptr);

  std::vector<char> data(1024, 'A');
  ASSERT_EQ(hipMemcpyAsync(gpu_ptr, data.data(), 1024,
                           hipMemcpyHostToDevice, stream_), hipSuccess);
  cuda_copy = true;

  if (cuda_copy) {
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
  }
}

#endif  // TRITON_ENABLE_ROCM

}  // namespace test
}}}  // namespace triton::backend::python

