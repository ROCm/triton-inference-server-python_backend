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
// Comprehensive ROCm Integration Tests
// Tests HIP stream operations, memory transfers, and synchronization
// that are critical for the CUDA-to-ROCm port.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

#include "pb_utils.h"
#include "pb_memory.h"
#include "pb_exception.h"
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {
namespace test {

//
// HIP Stream Tests - Critical for async operations in the backend
//
class HIPStreamTest : public ::testing::Test {
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
    for (hipStream_t stream : streams_) {
      (void)hipStreamDestroy(stream);
    }
    streams_.clear();
    for (void* ptr : allocations_) {
      (void)hipFree(ptr);
    }
    allocations_.clear();
  }

  hipStream_t CreateStream()
  {
    hipStream_t stream;
    hipError_t err = hipStreamCreate(&stream);
    if (err == hipSuccess) {
      streams_.push_back(stream);
    }
    return stream;
  }

  void* AllocateGPU(size_t size)
  {
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size) == hipSuccess) {
      allocations_.push_back(ptr);
    }
    return ptr;
  }

  std::vector<hipStream_t> streams_;
  std::vector<void*> allocations_;
};

// Test stream creation - used in pb_stub.cc for DLPack proxy streams
TEST_F(HIPStreamTest, StreamCreation)
{
  hipStream_t stream = CreateStream();
  EXPECT_NE(stream, nullptr);
}

// Test multiple stream creation
TEST_F(HIPStreamTest, MultipleStreams)
{
  const int num_streams = 4;
  std::vector<hipStream_t> test_streams;

  for (int i = 0; i < num_streams; ++i) {
    hipStream_t stream = CreateStream();
    ASSERT_NE(stream, nullptr);
    test_streams.push_back(stream);
  }

  // All streams should be unique
  for (size_t i = 0; i < test_streams.size(); ++i) {
    for (size_t j = i + 1; j < test_streams.size(); ++j) {
      EXPECT_NE(test_streams[i], test_streams[j]);
    }
  }
}

// Test stream synchronization - critical for infer_response.cc
TEST_F(HIPStreamTest, StreamSynchronize)
{
  hipStream_t stream = CreateStream();
  ASSERT_NE(stream, nullptr);

  // Allocate memory
  size_t size = 1024 * 1024;  // 1MB
  void* gpu_ptr = AllocateGPU(size);
  ASSERT_NE(gpu_ptr, nullptr);

  // Launch async memset
  ASSERT_EQ(hipMemsetAsync(gpu_ptr, 0xAB, size, stream), hipSuccess);

  // Synchronize stream
  EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
}

// Test async memory copy with stream - used throughout the backend
TEST_F(HIPStreamTest, AsyncMemoryCopy)
{
  hipStream_t stream = CreateStream();
  ASSERT_NE(stream, nullptr);

  size_t size = 4096;
  std::vector<float> host_src(size / sizeof(float), 3.14f);
  std::vector<float> host_dst(size / sizeof(float), 0.0f);

  void* gpu_ptr = AllocateGPU(size);
  ASSERT_NE(gpu_ptr, nullptr);

  // Async H2D copy
  ASSERT_EQ(hipMemcpyAsync(gpu_ptr, host_src.data(), size,
                           hipMemcpyHostToDevice, stream), hipSuccess);

  // Async D2H copy
  ASSERT_EQ(hipMemcpyAsync(host_dst.data(), gpu_ptr, size,
                           hipMemcpyDeviceToHost, stream), hipSuccess);

  // Synchronize
  ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

  // Verify data
  for (size_t i = 0; i < host_dst.size(); ++i) {
    EXPECT_FLOAT_EQ(host_dst[i], 3.14f);
  }
}

// Test default stream (stream 0) synchronization
TEST_F(HIPStreamTest, DefaultStreamSync)
{
  size_t size = 1024;
  void* gpu_ptr = AllocateGPU(size);
  ASSERT_NE(gpu_ptr, nullptr);

  std::vector<char> host_data(size, 0x55);

  // Use default stream (0)
  ASSERT_EQ(hipMemcpy(gpu_ptr, host_data.data(), size,
                      hipMemcpyHostToDevice), hipSuccess);

  // Synchronize default stream - used in pb_memory.cc for D2D copies
  EXPECT_EQ(hipStreamSynchronize(0), hipSuccess);
}

//
// HIP Memory Transfer Tests
//
class HIPMemoryTransferTest : public ::testing::Test {
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
    for (auto& alloc : allocations_) {
      (void)hipSetDevice(alloc.device);
      (void)hipFree(alloc.ptr);
    }
    allocations_.clear();
  }

  struct Allocation {
    void* ptr;
    int device;
  };

  void* AllocateOnDevice(size_t size, int device)
  {
    (void)hipSetDevice(device);
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size) == hipSuccess) {
      allocations_.push_back({ptr, device});
    }
    return ptr;
  }

  int device_count_ = 0;
  std::vector<Allocation> allocations_;
};

// Test hipMemcpy kinds - all used in pb_memory.cc CopyBuffer
TEST_F(HIPMemoryTransferTest, MemcpyHostToDevice)
{
  size_t size = 1024;
  std::vector<int> host_data(size / sizeof(int), 42);

  void* gpu_ptr = AllocateOnDevice(size, 0);
  ASSERT_NE(gpu_ptr, nullptr);

  EXPECT_EQ(hipMemcpy(gpu_ptr, host_data.data(), size,
                      hipMemcpyHostToDevice), hipSuccess);
}

TEST_F(HIPMemoryTransferTest, MemcpyDeviceToHost)
{
  size_t size = 1024;
  void* gpu_ptr = AllocateOnDevice(size, 0);
  ASSERT_NE(gpu_ptr, nullptr);

  // Initialize GPU memory
  ASSERT_EQ(hipMemset(gpu_ptr, 0xCD, size), hipSuccess);

  std::vector<char> host_data(size);
  EXPECT_EQ(hipMemcpy(host_data.data(), gpu_ptr, size,
                      hipMemcpyDeviceToHost), hipSuccess);

  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(static_cast<unsigned char>(host_data[i]), 0xCD);
  }
}

TEST_F(HIPMemoryTransferTest, MemcpyDeviceToDevice)
{
  size_t size = 1024;
  void* src_ptr = AllocateOnDevice(size, 0);
  void* dst_ptr = AllocateOnDevice(size, 0);
  ASSERT_NE(src_ptr, nullptr);
  ASSERT_NE(dst_ptr, nullptr);

  // Initialize source
  ASSERT_EQ(hipMemset(src_ptr, 0xEF, size), hipSuccess);

  // D2D copy on same device
  EXPECT_EQ(hipMemcpy(dst_ptr, src_ptr, size,
                      hipMemcpyDeviceToDevice), hipSuccess);

  // Verify
  std::vector<char> host_data(size);
  ASSERT_EQ(hipMemcpy(host_data.data(), dst_ptr, size,
                      hipMemcpyDeviceToHost), hipSuccess);

  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(static_cast<unsigned char>(host_data[i]), 0xEF);
  }
}

// Test hipMemcpyPeer for cross-device transfers (used in pb_memory.cc)
TEST_F(HIPMemoryTransferTest, MemcpyPeerMultiGPU)
{
  if (device_count_ < 2) {
    GTEST_SKIP() << "Multi-GPU test requires at least 2 GPUs";
  }

  size_t size = 4096;
  void* src_ptr = AllocateOnDevice(size, 0);
  void* dst_ptr = AllocateOnDevice(size, 1);
  ASSERT_NE(src_ptr, nullptr);
  ASSERT_NE(dst_ptr, nullptr);

  // Initialize source on device 0
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  std::vector<float> pattern(size / sizeof(float), 2.718f);
  ASSERT_EQ(hipMemcpy(src_ptr, pattern.data(), size,
                      hipMemcpyHostToDevice), hipSuccess);

  // Peer copy from device 0 to device 1
  EXPECT_EQ(hipMemcpyPeer(dst_ptr, 1, src_ptr, 0, size), hipSuccess);

  // Verify on device 1
  std::vector<float> verify(size / sizeof(float));
  ASSERT_EQ(hipSetDevice(1), hipSuccess);
  ASSERT_EQ(hipMemcpy(verify.data(), dst_ptr, size,
                      hipMemcpyDeviceToHost), hipSuccess);

  for (size_t i = 0; i < verify.size(); ++i) {
    EXPECT_FLOAT_EQ(verify[i], 2.718f);
  }
}

//
// HIP IPC Handle Tests
//
class HIPIpcTest : public ::testing::Test {
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

// Test hipIpcGetMemHandle - used in pb_memory.cc and stub_launcher.cc
TEST_F(HIPIpcTest, GetMemHandle)
{
  size_t size = 4096;
  void* gpu_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&gpu_ptr, size), hipSuccess);
  allocations_.push_back(gpu_ptr);

  hipIpcMemHandle_t handle;
  hipError_t err = hipIpcGetMemHandle(&handle, gpu_ptr);

  // IPC may not be supported in all environments
  if (err == hipErrorNotSupported) {
    GTEST_SKIP() << "IPC not supported in this environment";
  }

  EXPECT_EQ(err, hipSuccess);
}

// Test that IPC handle has non-zero content
TEST_F(HIPIpcTest, HandleHasContent)
{
  size_t size = 4096;
  void* gpu_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&gpu_ptr, size), hipSuccess);
  allocations_.push_back(gpu_ptr);

  hipIpcMemHandle_t handle;
  memset(&handle, 0, sizeof(handle));

  hipError_t err = hipIpcGetMemHandle(&handle, gpu_ptr);
  if (err == hipErrorNotSupported) {
    GTEST_SKIP() << "IPC not supported";
  }
  ASSERT_EQ(err, hipSuccess);

  // Handle should have some non-zero bytes
  bool has_content = false;
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&handle);
  for (size_t i = 0; i < sizeof(handle); ++i) {
    if (bytes[i] != 0) {
      has_content = true;
      break;
    }
  }
  EXPECT_TRUE(has_content);
}

//
// HIP Pointer Attribute Tests
//
class HIPPointerAttributeTest : public ::testing::Test {
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

// Test hipPointerGetAttributes - used in HIPHandler
TEST_F(HIPPointerAttributeTest, GetAttributesGPUPointer)
{
  size_t size = 4096;
  void* gpu_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&gpu_ptr, size), hipSuccess);
  allocations_.push_back(gpu_ptr);

  hipPointerAttribute_t attrs;
  EXPECT_EQ(hipPointerGetAttributes(&attrs, gpu_ptr), hipSuccess);

  EXPECT_EQ(attrs.type, hipMemoryTypeDevice);
  EXPECT_EQ(attrs.device, 0);
}

// Test pointer attributes for offset pointer
TEST_F(HIPPointerAttributeTest, GetAttributesOffsetPointer)
{
  size_t size = 8192;
  void* gpu_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&gpu_ptr, size), hipSuccess);
  allocations_.push_back(gpu_ptr);

  // Get attributes for pointer with offset
  void* offset_ptr = static_cast<char*>(gpu_ptr) + 1024;

  hipPointerAttribute_t attrs;
  EXPECT_EQ(hipPointerGetAttributes(&attrs, offset_ptr), hipSuccess);
  EXPECT_EQ(attrs.type, hipMemoryTypeDevice);
}

//
// HIPHandler Integration with Shared Memory Tests
//
class HIPHandlerShmIntegrationTest : public ::testing::Test {
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

    shm_region_name_ = "/hip_shm_integration_" + std::to_string(getpid());
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

  HIPHandler* hip_handler_;
  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
  std::vector<void*> allocations_;
};

// Test storing hipIpcMemHandle_t in shared memory (used for GPU tensor transfer)
TEST_F(HIPHandlerShmIntegrationTest, StoreIpcHandleInShm)
{
  size_t gpu_size = 4096;
  void* gpu_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&gpu_ptr, gpu_size), hipSuccess);
  allocations_.push_back(gpu_ptr);

  hipIpcMemHandle_t handle;
  hipError_t err = hipIpcGetMemHandle(&handle, gpu_ptr);
  if (err == hipErrorNotSupported) {
    GTEST_SKIP() << "IPC not supported";
  }
  ASSERT_EQ(err, hipSuccess);

  // Store handle in shared memory
  auto shm_alloc = shm_pool_->Construct<hipIpcMemHandle_t>();
  *shm_alloc.data_ = handle;

  // Load and verify
  auto loaded = shm_pool_->Load<hipIpcMemHandle_t>(shm_alloc.handle_);
  EXPECT_EQ(memcmp(loaded.data_.get(), &handle, sizeof(handle)), 0);
}

// Test PbMemory GPU pointer offset calculation
TEST_F(HIPHandlerShmIntegrationTest, GPUPointerOffsetCalculation)
{
  size_t size = 8192;
  void* gpu_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&gpu_ptr, size), hipSuccess);
  allocations_.push_back(gpu_ptr);

  // Get base address through HIPHandler
  hipDeviceptr_t start_address = 0;
  EXPECT_NO_THROW(hip_handler_->PointerGetAttribute(
      &start_address, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
      reinterpret_cast<hipDeviceptr_t>(gpu_ptr)));

  // Base should be <= allocated pointer
  EXPECT_LE(reinterpret_cast<uintptr_t>(start_address),
            reinterpret_cast<uintptr_t>(gpu_ptr));

  // Offset should be >= 0
  uintptr_t offset = reinterpret_cast<uintptr_t>(gpu_ptr) -
                     reinterpret_cast<uintptr_t>(start_address);
  EXPECT_GE(offset, 0);
}

//
// Concurrent GPU Operations Test
//
class ConcurrentGPUTest : public ::testing::Test {
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
  std::mutex alloc_mutex_;

  void* ThreadSafeAlloc(size_t size)
  {
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size) == hipSuccess) {
      std::lock_guard<std::mutex> lock(alloc_mutex_);
      allocations_.push_back(ptr);
    }
    return ptr;
  }
};

// Test concurrent GPU memory allocations
TEST_F(ConcurrentGPUTest, ConcurrentAllocations)
{
  const int num_threads = 4;
  const int allocs_per_thread = 10;
  std::atomic<int> success_count{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([this, &success_count, allocs_per_thread]() {
      for (int i = 0; i < allocs_per_thread; ++i) {
        void* ptr = ThreadSafeAlloc(1024);
        if (ptr != nullptr) {
          success_count++;
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(success_count.load(), num_threads * allocs_per_thread);
}

// Test concurrent stream operations
TEST_F(ConcurrentGPUTest, ConcurrentStreamOperations)
{
  const int num_threads = 4;
  std::atomic<int> success_count{0};
  std::vector<hipStream_t> streams(num_threads);

  // Create streams
  for (int i = 0; i < num_threads; ++i) {
    ASSERT_EQ(hipStreamCreate(&streams[i]), hipSuccess);
  }

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([this, &success_count, &streams, t]() {
      void* ptr = ThreadSafeAlloc(1024);
      if (ptr != nullptr) {
        if (hipMemsetAsync(ptr, t, 1024, streams[t]) == hipSuccess &&
            hipStreamSynchronize(streams[t]) == hipSuccess) {
          success_count++;
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Cleanup streams
  for (auto& stream : streams) {
    (void)hipStreamDestroy(stream);
  }

  EXPECT_EQ(success_count.load(), num_threads);
}

}  // namespace test
}}}  // namespace triton::backend::python

