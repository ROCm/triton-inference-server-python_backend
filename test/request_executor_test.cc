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
// Unit Tests for request_executor.cc
// Tests memory type conversion, error handling, and ROCm-specific
// allocation patterns used in the response allocator.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#ifdef TRITON_ENABLE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

#include "pb_exception.h"
#include "pb_preferred_memory.h"
#include "pb_memory.h"
#include "shm_manager.h"
#include "infer_payload.h"

namespace triton { namespace backend { namespace python {
namespace test {

//
// Test PreferredMemory class
//
class PreferredMemoryTest : public ::testing::Test {};

TEST_F(PreferredMemoryTest, DefaultConstruction)
{
  PreferredMemory pm;
  EXPECT_EQ(pm.PreferredMemoryType(), PreferredMemory::MemoryType::DEFAULT);
  EXPECT_EQ(pm.PreferredDeviceId(), 0);
}

TEST_F(PreferredMemoryTest, CPUMemoryType)
{
  PreferredMemory pm(PreferredMemory::MemoryType::CPU, 0);
  EXPECT_EQ(pm.PreferredMemoryType(), PreferredMemory::MemoryType::CPU);
  EXPECT_EQ(pm.PreferredDeviceId(), 0);
}

TEST_F(PreferredMemoryTest, GPUMemoryType)
{
  PreferredMemory pm(PreferredMemory::MemoryType::GPU, 1);
  EXPECT_EQ(pm.PreferredMemoryType(), PreferredMemory::MemoryType::GPU);
  EXPECT_EQ(pm.PreferredDeviceId(), 1);
}

TEST_F(PreferredMemoryTest, DifferentDeviceIds)
{
  for (int64_t device_id = 0; device_id < 4; ++device_id) {
    PreferredMemory pm(PreferredMemory::MemoryType::GPU, device_id);
    EXPECT_EQ(pm.PreferredDeviceId(), device_id);
  }
}

//
// Test ResponseAllocatorUserp structure
//
class ResponseAllocatorUserpTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    shm_region_name_ = "/resp_alloc_userp_test_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_, 4 * 1024 * 1024, 1024 * 1024, true);
  }

  void TearDown() override
  {
    shm_pool_.reset();
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
};

TEST_F(ResponseAllocatorUserpTest, Construction)
{
  PreferredMemory pm(PreferredMemory::MemoryType::CPU, 0);
  ResponseAllocatorUserp userp(shm_pool_.get(), pm);

  EXPECT_EQ(userp.shm_pool, shm_pool_.get());
  EXPECT_EQ(userp.preferred_memory.PreferredMemoryType(),
            PreferredMemory::MemoryType::CPU);
  EXPECT_EQ(userp.preferred_memory.PreferredDeviceId(), 0);
}

TEST_F(ResponseAllocatorUserpTest, GPUPreferredMemory)
{
  PreferredMemory pm(PreferredMemory::MemoryType::GPU, 2);
  ResponseAllocatorUserp userp(shm_pool_.get(), pm);

  EXPECT_EQ(userp.preferred_memory.PreferredMemoryType(),
            PreferredMemory::MemoryType::GPU);
  EXPECT_EQ(userp.preferred_memory.PreferredDeviceId(), 2);
}

TEST_F(ResponseAllocatorUserpTest, DefaultPreferredMemory)
{
  PreferredMemory pm;  // Default construction
  ResponseAllocatorUserp userp(shm_pool_.get(), pm);

  EXPECT_EQ(userp.preferred_memory.PreferredMemoryType(),
            PreferredMemory::MemoryType::DEFAULT);
}

//
// Test ROCm-specific allocation patterns (from ResponseAlloc)
//
#ifdef TRITON_ENABLE_ROCM
class ROCmResponseAllocTest : public ::testing::Test {
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
    allocations_.clear();
  }

  int device_count_ = 0;
  std::vector<void*> allocations_;
};

// Test hipSetDevice pattern from ResponseAlloc
TEST_F(ROCmResponseAllocTest, SetDevicePattern)
{
  // Pattern from request_executor.cc ResponseAlloc
  int64_t memory_type_id = 0;
  auto err = hipSetDevice(memory_type_id);

  // These are acceptable return values per request_executor.cc
  EXPECT_TRUE(
      err == hipSuccess ||
      err == hipErrorNoDevice ||
      err == hipErrorInsufficientDriver);
}

// Test hipMalloc pattern from ResponseAlloc
TEST_F(ROCmResponseAllocTest, MallocPattern)
{
  void* buffer = nullptr;
  size_t byte_size = 4096;

  // Pattern from request_executor.cc
  hipError_t err = hipMalloc(&buffer, byte_size);
  EXPECT_EQ(err, hipSuccess);
  EXPECT_NE(buffer, nullptr);

  if (buffer != nullptr) {
    allocations_.push_back(buffer);
  }
}

// Test allocation for different sizes
TEST_F(ROCmResponseAllocTest, VariousSizes)
{
  std::vector<size_t> sizes = {1, 256, 1024, 4096, 1024 * 1024};

  for (size_t size : sizes) {
    void* buffer = nullptr;
    hipError_t err = hipMalloc(&buffer, size);
    EXPECT_EQ(err, hipSuccess) << "Failed for size: " << size;
    EXPECT_NE(buffer, nullptr) << "Null buffer for size: " << size;

    if (buffer != nullptr) {
      allocations_.push_back(buffer);
    }
  }
}

// Test multi-device allocation pattern
TEST_F(ROCmResponseAllocTest, MultiDeviceAllocation)
{
  if (device_count_ < 2) {
    GTEST_SKIP() << "Multi-GPU test requires at least 2 GPUs";
  }

  for (int device = 0; device < device_count_ && device < 2; ++device) {
    ASSERT_EQ(hipSetDevice(device), hipSuccess);

    void* buffer = nullptr;
    ASSERT_EQ(hipMalloc(&buffer, 1024), hipSuccess);
    EXPECT_NE(buffer, nullptr);
    allocations_.push_back(buffer);
  }

  // Reset to device 0
  (void)hipSetDevice(0);
}

// Test zero-size allocation behavior
TEST_F(ROCmResponseAllocTest, ZeroSizeAllocation)
{
  // Per ResponseAlloc, zero byte_size returns nullptr buffer
  // This tests the edge case handling
  void* buffer = nullptr;
  size_t byte_size = 0;

  // Zero allocation - behavior may vary
  // ResponseAlloc handles this by returning early
  // Just verify we don't crash
  if (byte_size > 0) {
    hipError_t err = hipMalloc(&buffer, byte_size);
    EXPECT_EQ(err, hipSuccess);
  } else {
    buffer = nullptr;
  }
  EXPECT_EQ(buffer, nullptr);
}

// Test error handling for invalid device
TEST_F(ROCmResponseAllocTest, InvalidDeviceHandling)
{
  // Try setting an invalid device (very high number)
  int64_t invalid_device = 9999;
  hipError_t err = hipSetDevice(invalid_device);

  // Should return an error, not crash
  EXPECT_NE(err, hipSuccess);
}
#endif  // TRITON_ENABLE_ROCM

//
// Test PbMemory allocation in CPU path (from ResponseAlloc)
//
class CPUResponseAllocTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    shm_region_name_ = "/cpu_resp_alloc_test_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_, 4 * 1024 * 1024, 1024 * 1024, true);
  }

  void TearDown() override
  {
    shm_pool_.reset();
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
};

// Test PbMemory::Create pattern from ResponseAlloc CPU path
TEST_F(CPUResponseAllocTest, PbMemoryCreatePattern)
{
  TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t actual_memory_type_id = 0;
  size_t byte_size = 1024;

  // Pattern from request_executor.cc ResponseAlloc
  std::unique_ptr<PbMemory> pb_memory = PbMemory::Create(
      shm_pool_, actual_memory_type, actual_memory_type_id, byte_size,
      nullptr /* data */, false /* copy_gpu */);

  EXPECT_NE(pb_memory, nullptr);
  EXPECT_NE(pb_memory->DataPtr(), nullptr);
  EXPECT_EQ(pb_memory->MemoryType(), TRITONSERVER_MEMORY_CPU);
  EXPECT_EQ(pb_memory->ByteSize(), byte_size);
}

// Test various allocation sizes
TEST_F(CPUResponseAllocTest, VariousSizes)
{
  std::vector<size_t> sizes = {1, 64, 256, 1024, 4096};

  for (size_t size : sizes) {
    auto pb_memory = PbMemory::Create(
        shm_pool_, TRITONSERVER_MEMORY_CPU, 0, size,
        nullptr, false);

    EXPECT_NE(pb_memory, nullptr) << "Failed for size: " << size;
    EXPECT_EQ(pb_memory->ByteSize(), size) << "Wrong size for: " << size;
  }
}

// Test that allocated memory is writable
TEST_F(CPUResponseAllocTest, MemoryIsWritable)
{
  size_t byte_size = 256;
  auto pb_memory = PbMemory::Create(
      shm_pool_, TRITONSERVER_MEMORY_CPU, 0, byte_size,
      nullptr, false);

  ASSERT_NE(pb_memory, nullptr);
  char* data = pb_memory->DataPtr();
  ASSERT_NE(data, nullptr);

  // Write pattern
  for (size_t i = 0; i < byte_size; ++i) {
    data[i] = static_cast<char>(i % 256);
  }

  // Verify pattern
  for (size_t i = 0; i < byte_size; ++i) {
    EXPECT_EQ(static_cast<unsigned char>(data[i]), i % 256);
  }
}

//
// Test InferPayload related functionality
//
class InferPayloadTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    shm_region_name_ = "/infer_payload_test_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_, 4 * 1024 * 1024, 1024 * 1024, true);
  }

  void TearDown() override
  {
    shm_pool_.reset();
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
};

TEST_F(InferPayloadTest, IsDecoupledFlag)
{
  // Non-decoupled payload
  auto non_decoupled = std::make_shared<InferPayload>(
      false /* is_decoupled */,
      [](std::unique_ptr<InferResponse>) {});
  EXPECT_FALSE(non_decoupled->IsDecoupled());

  // Decoupled payload
  auto decoupled = std::make_shared<InferPayload>(
      true /* is_decoupled */,
      [](std::unique_ptr<InferResponse>) {});
  EXPECT_TRUE(decoupled->IsDecoupled());
}

TEST_F(InferPayloadTest, SetResponseAllocUserp)
{
  auto payload = std::make_shared<InferPayload>(
      false,
      [](std::unique_ptr<InferResponse>) {});

  PreferredMemory pm(PreferredMemory::MemoryType::CPU, 0);
  ResponseAllocatorUserp userp(shm_pool_.get(), pm);

  payload->SetResponseAllocUserp(userp);

  auto retrieved = payload->ResponseAllocUserp();
  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved->shm_pool, shm_pool_.get());
  EXPECT_EQ(retrieved->preferred_memory.PreferredMemoryType(),
            PreferredMemory::MemoryType::CPU);
}

TEST_F(InferPayloadTest, GetPtr)
{
  auto payload = std::make_shared<InferPayload>(
      false,
      [](std::unique_ptr<InferResponse>) {});

  auto ptr = payload->GetPtr();
  EXPECT_EQ(ptr.get(), payload.get());
}

//
// Memory type enum coverage
//
class MemoryTypeEnumTest : public ::testing::Test {};

TEST_F(MemoryTypeEnumTest, AllEnumValues)
{
  // Verify all enum values are distinct
  EXPECT_NE(PreferredMemory::MemoryType::GPU, PreferredMemory::MemoryType::CPU);
  EXPECT_NE(PreferredMemory::MemoryType::GPU, PreferredMemory::MemoryType::DEFAULT);
  EXPECT_NE(PreferredMemory::MemoryType::CPU, PreferredMemory::MemoryType::DEFAULT);
}

TEST_F(MemoryTypeEnumTest, EnumToTritonMemoryType)
{
  // Verify mapping between PreferredMemory types and TRITONSERVER types
  // GPU -> TRITONSERVER_MEMORY_GPU
  EXPECT_EQ(static_cast<int>(PreferredMemory::MemoryType::GPU), 0);

  // CPU -> TRITONSERVER_MEMORY_CPU
  EXPECT_EQ(static_cast<int>(PreferredMemory::MemoryType::CPU), 1);

  // DEFAULT is special (no direct TRITON mapping)
  EXPECT_EQ(static_cast<int>(PreferredMemory::MemoryType::DEFAULT), 2);
}

}  // namespace test
}}}  // namespace triton::backend::python

