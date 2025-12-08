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

#include <cstring>
#include <memory>
#include <string>

#include "pb_memory.h"
#include "pb_exception.h"
#include "shm_manager.h"

#ifdef TRITON_ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace triton { namespace backend { namespace python {
namespace test {

//
// Test Fixture for CPU Memory Tests
//
class PbMemoryCPUTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Create a shared memory manager for testing
    shm_region_name_ = "/pb_memory_test_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_,
        1024 * 1024,  // 1MB initial size
        512 * 1024,   // 512KB growth
        true          // create
    );
  }

  void TearDown() override
  {
    shm_pool_.reset();
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
};

// Test creating CPU memory with data
TEST_F(PbMemoryCPUTest, CreateWithData)
{
  const char* test_data = "Hello, PbMemory!";
  size_t data_size = strlen(test_data) + 1;

  auto pb_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,  // memory_type_id
      data_size,
      const_cast<char*>(test_data));

  ASSERT_NE(pb_memory, nullptr);
  EXPECT_EQ(pb_memory->MemoryType(), TRITONSERVER_MEMORY_CPU);
  EXPECT_EQ(pb_memory->MemoryTypeId(), 0);
  EXPECT_EQ(pb_memory->ByteSize(), data_size);
  EXPECT_STREQ(pb_memory->DataPtr(), test_data);
}

// Test creating CPU memory with nullptr
TEST_F(PbMemoryCPUTest, CreateWithNullptr)
{
  auto pb_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,
      1024,
      nullptr);

  ASSERT_NE(pb_memory, nullptr);
  EXPECT_EQ(pb_memory->MemoryType(), TRITONSERVER_MEMORY_CPU);
  EXPECT_EQ(pb_memory->ByteSize(), 1024);
  EXPECT_NE(pb_memory->DataPtr(), nullptr);
}

// Test ShmStructSize calculation for CPU memory
TEST_F(PbMemoryCPUTest, ShmStructSizeCPU)
{
  uint64_t byte_size = 1024;
  uint64_t struct_size =
      PbMemory::ShmStructSize(TRITONSERVER_MEMORY_CPU, byte_size);

  // For CPU memory, struct size should include MemoryShm + data
  EXPECT_EQ(struct_size, sizeof(MemoryShm) + byte_size);
}

// Test CopyBuffer between CPU buffers
TEST_F(PbMemoryCPUTest, CopyBufferCPUtoCPU)
{
  const char* src_data = "Source data for copy test";
  size_t data_size = strlen(src_data) + 1;

  auto src_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,
      data_size,
      const_cast<char*>(src_data));

  auto dst_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,
      data_size,
      nullptr);

  // Perform copy
  PbMemory::CopyBuffer(dst_memory, src_memory);

  // Verify data was copied
  EXPECT_STREQ(dst_memory->DataPtr(), src_data);
}

// Test CopyBuffer with mismatched sizes throws
TEST_F(PbMemoryCPUTest, CopyBufferSizeMismatchThrows)
{
  auto src_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,
      100,
      nullptr);

  auto dst_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,
      200,  // Different size
      nullptr);

  EXPECT_THROW(
      PbMemory::CopyBuffer(dst_memory, src_memory),
      PythonBackendException);
}

// Test memory release ID functionality
TEST_F(PbMemoryCPUTest, MemoryReleaseId)
{
  auto pb_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,
      1024,
      nullptr);

  // Initial release ID should be 0
  EXPECT_EQ(pb_memory->MemoryReleaseId(), 0);

  // Set and verify release ID
  uint64_t release_id = 12345;
  pb_memory->SetMemoryReleaseId(release_id);
  EXPECT_EQ(pb_memory->MemoryReleaseId(), release_id);
}

// Test ShmHandle returns valid handle
TEST_F(PbMemoryCPUTest, ShmHandleValid)
{
  auto pb_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,
      1024,
      nullptr);

  auto handle = pb_memory->ShmHandle();
  EXPECT_NE(handle, 0);
}

// Test LoadFromSharedMemory for CPU memory
TEST_F(PbMemoryCPUTest, LoadFromSharedMemory)
{
  const char* test_data = "Shared memory test data";
  size_t data_size = strlen(test_data) + 1;

  // Create memory and get handle
  auto original = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,
      data_size,
      const_cast<char*>(test_data));

  auto handle = original->ShmHandle();

  // Load from shared memory using handle
  auto loaded = PbMemory::LoadFromSharedMemory(
      shm_pool_,
      handle,
      false  // open_cuda_handle
  );

  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->MemoryType(), TRITONSERVER_MEMORY_CPU);
  EXPECT_EQ(loaded->ByteSize(), data_size);
  EXPECT_STREQ(loaded->DataPtr(), test_data);
}

// Test memory release callback
TEST_F(PbMemoryCPUTest, MemoryReleaseCallback)
{
  bool callback_called = false;

  {
    auto pb_memory = PbMemory::Create(
        shm_pool_,
        TRITONSERVER_MEMORY_CPU,
        0,
        1024,
        nullptr);

    pb_memory->SetMemoryReleaseCallback([&callback_called]() {
      callback_called = true;
    });

    // Callback should not be called yet
    EXPECT_FALSE(callback_called);
  }

  // Callback should be called when memory is destroyed
  EXPECT_TRUE(callback_called);
}

// Test setting release callback twice throws
TEST_F(PbMemoryCPUTest, SetReleaseCallbackTwiceThrows)
{
  auto pb_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,
      1024,
      nullptr);

  pb_memory->SetMemoryReleaseCallback([]() {});

  EXPECT_THROW(
      pb_memory->SetMemoryReleaseCallback([]() {}),
      PythonBackendException);
}

#ifdef TRITON_ENABLE_ROCM

//
// Test Fixture for GPU Memory Tests (ROCm-specific)
//
class PbMemoryGPUTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Check if HIP is available
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }

    // Create shared memory manager
    shm_region_name_ = "/pb_memory_gpu_test_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_,
        1024 * 1024,
        512 * 1024,
        true);
  }

  void TearDown() override
  {
    // Free any allocated GPU memory
    for (void* ptr : gpu_allocations_) {
      (void)hipFree(ptr);
    }
    gpu_allocations_.clear();
    shm_pool_.reset();
  }

  void* AllocateGPUMemory(size_t size)
  {
    void* ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, size);
    if (err == hipSuccess && ptr != nullptr) {
      gpu_allocations_.push_back(ptr);
    }
    return ptr;
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
  std::vector<void*> gpu_allocations_;
};

// Test ShmStructSize calculation for GPU memory
TEST_F(PbMemoryGPUTest, ShmStructSizeGPU)
{
  uint64_t byte_size = 1024;
  uint64_t struct_size =
      PbMemory::ShmStructSize(TRITONSERVER_MEMORY_GPU, byte_size);

  // For GPU memory, struct size should include MemoryShm + IPC handle
  EXPECT_EQ(struct_size, sizeof(MemoryShm) + sizeof(hipIpcMemHandle_t));
}

// Test creating GPU memory
TEST_F(PbMemoryGPUTest, CreateGPUMemory)
{
  size_t data_size = 4096;
  void* gpu_ptr = AllocateGPUMemory(data_size);
  ASSERT_NE(gpu_ptr, nullptr) << "Failed to allocate GPU memory";

  auto pb_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_GPU,
      0,  // device 0
      data_size,
      static_cast<char*>(gpu_ptr));

  ASSERT_NE(pb_memory, nullptr);
  EXPECT_EQ(pb_memory->MemoryType(), TRITONSERVER_MEMORY_GPU);
  EXPECT_EQ(pb_memory->MemoryTypeId(), 0);
  EXPECT_EQ(pb_memory->ByteSize(), data_size);
}

// Test CopyBuffer from CPU to GPU
TEST_F(PbMemoryGPUTest, CopyBufferCPUtoGPU)
{
  size_t data_size = 1024;
  std::vector<char> src_data(data_size);
  std::fill(src_data.begin(), src_data.end(), 0xAB);

  auto src_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,
      data_size,
      src_data.data());

  void* gpu_ptr = AllocateGPUMemory(data_size);
  ASSERT_NE(gpu_ptr, nullptr);

  auto dst_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_GPU,
      0,
      data_size,
      static_cast<char*>(gpu_ptr),
      false  // don't copy to GPU yet
  );

  // Perform copy
  PbMemory::CopyBuffer(dst_memory, src_memory);

  // Verify by copying back to host
  std::vector<char> verify_data(data_size);
  hipError_t err = hipMemcpy(
      verify_data.data(), gpu_ptr, data_size, hipMemcpyDeviceToHost);
  ASSERT_EQ(err, hipSuccess);
  EXPECT_EQ(verify_data, src_data);
}

// Test CopyBuffer from GPU to CPU
TEST_F(PbMemoryGPUTest, CopyBufferGPUtoCPU)
{
  size_t data_size = 1024;

  // Create GPU memory with test pattern
  void* gpu_ptr = AllocateGPUMemory(data_size);
  ASSERT_NE(gpu_ptr, nullptr);

  std::vector<char> pattern(data_size);
  std::fill(pattern.begin(), pattern.end(), 0xCD);
  hipError_t err = hipMemcpy(
      gpu_ptr, pattern.data(), data_size, hipMemcpyHostToDevice);
  ASSERT_EQ(err, hipSuccess);

  auto src_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_GPU,
      0,
      data_size,
      static_cast<char*>(gpu_ptr));

  auto dst_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_CPU,
      0,
      data_size,
      nullptr);

  // Perform copy
  PbMemory::CopyBuffer(dst_memory, src_memory);

  // Verify
  for (size_t i = 0; i < data_size; ++i) {
    EXPECT_EQ(static_cast<unsigned char>(dst_memory->DataPtr()[i]), 0xCD)
        << "Mismatch at index " << i;
  }
}

// Test CopyBuffer from GPU to GPU (same device)
TEST_F(PbMemoryGPUTest, CopyBufferGPUtoGPUSameDevice)
{
  size_t data_size = 1024;

  // Create source GPU memory with test pattern
  void* src_gpu_ptr = AllocateGPUMemory(data_size);
  void* dst_gpu_ptr = AllocateGPUMemory(data_size);
  ASSERT_NE(src_gpu_ptr, nullptr);
  ASSERT_NE(dst_gpu_ptr, nullptr);

  std::vector<char> pattern(data_size);
  std::fill(pattern.begin(), pattern.end(), 0xEF);
  hipError_t err = hipMemcpy(
      src_gpu_ptr, pattern.data(), data_size, hipMemcpyHostToDevice);
  ASSERT_EQ(err, hipSuccess);

  auto src_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_GPU,
      0,
      data_size,
      static_cast<char*>(src_gpu_ptr));

  auto dst_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_GPU,
      0,
      data_size,
      static_cast<char*>(dst_gpu_ptr),
      false);

  // Perform copy
  PbMemory::CopyBuffer(dst_memory, src_memory);

  // Verify by copying back to host
  std::vector<char> verify_data(data_size);
  err = hipMemcpy(
      verify_data.data(), dst_gpu_ptr, data_size, hipMemcpyDeviceToHost);
  ASSERT_EQ(err, hipSuccess);
  EXPECT_EQ(verify_data, pattern);
}

// Test SetHipIpcHandle
TEST_F(PbMemoryGPUTest, SetHipIpcHandle)
{
  size_t data_size = 4096;
  void* gpu_ptr = AllocateGPUMemory(data_size);
  ASSERT_NE(gpu_ptr, nullptr);

  // Get IPC handle for the GPU memory
  hipIpcMemHandle_t ipc_handle;
  hipError_t err = hipIpcGetMemHandle(&ipc_handle, gpu_ptr);
  if (err != hipSuccess) {
    GTEST_SKIP() << "IPC not supported in this environment";
  }

  auto pb_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_GPU,
      0,
      data_size,
      nullptr,
      false  // don't copy GPU
  );

  // Should be able to set the IPC handle
  EXPECT_NO_THROW(pb_memory->SetHipIpcHandle(&ipc_handle));
}

//
// Multi-GPU Memory Tests
//
class PbMemoryMultiGPUTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count < 2) {
      GTEST_SKIP() << "Multi-GPU tests require at least 2 GPUs";
    }

    device_count_ = device_count;

    shm_region_name_ = "/pb_memory_multi_gpu_test_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_,
        1024 * 1024,
        512 * 1024,
        true);
  }

  void TearDown() override
  {
    for (auto& alloc : gpu_allocations_) {
      (void)hipSetDevice(alloc.device);
      (void)hipFree(alloc.ptr);
    }
    gpu_allocations_.clear();
    shm_pool_.reset();
  }

  struct GPUAllocation {
    void* ptr;
    int device;
  };

  void* AllocateGPUMemoryOnDevice(size_t size, int device)
  {
    (void)hipSetDevice(device);
    void* ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, size);
    if (err == hipSuccess && ptr != nullptr) {
      gpu_allocations_.push_back({ptr, device});
    }
    return ptr;
  }

  int device_count_ = 0;
  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
  std::vector<GPUAllocation> gpu_allocations_;
};

// Test CopyBuffer between different GPUs
TEST_F(PbMemoryMultiGPUTest, CopyBufferGPUtoGPUDifferentDevices)
{
  size_t data_size = 1024;

  // Allocate on device 0
  void* src_gpu_ptr = AllocateGPUMemoryOnDevice(data_size, 0);
  ASSERT_NE(src_gpu_ptr, nullptr);

  // Allocate on device 1
  void* dst_gpu_ptr = AllocateGPUMemoryOnDevice(data_size, 1);
  ASSERT_NE(dst_gpu_ptr, nullptr);

  // Initialize source with pattern
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  std::vector<char> pattern(data_size);
  std::fill(pattern.begin(), pattern.end(), 0x12);
  hipError_t err = hipMemcpy(
      src_gpu_ptr, pattern.data(), data_size, hipMemcpyHostToDevice);
  ASSERT_EQ(err, hipSuccess);

  auto src_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_GPU,
      0,  // device 0
      data_size,
      static_cast<char*>(src_gpu_ptr));

  auto dst_memory = PbMemory::Create(
      shm_pool_,
      TRITONSERVER_MEMORY_GPU,
      1,  // device 1
      data_size,
      static_cast<char*>(dst_gpu_ptr),
      false);

  // This should use hipMemcpyPeer
  PbMemory::CopyBuffer(dst_memory, src_memory);

  // Verify
  ASSERT_EQ(hipSetDevice(1), hipSuccess);
  std::vector<char> verify_data(data_size);
  err = hipMemcpy(
      verify_data.data(), dst_gpu_ptr, data_size, hipMemcpyDeviceToHost);
  ASSERT_EQ(err, hipSuccess);
  EXPECT_EQ(verify_data, pattern);
}

#endif  // TRITON_ENABLE_ROCM

}  // namespace test
}}}  // namespace triton::backend::python

