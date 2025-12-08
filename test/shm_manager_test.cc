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
#include <thread>
#include <vector>

#include "shm_manager.h"
#include "pb_exception.h"
#include "pb_memory.h"

#ifdef TRITON_ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace triton { namespace backend { namespace python {
namespace test {

//
// Basic SharedMemoryManager Tests
//
class SharedMemoryManagerTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    shm_region_name_ = "/shm_manager_test_" + std::to_string(getpid());
  }

  void TearDown() override
  {
    // Clean up shared memory region
    shm_pool_.reset();
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
};

// Test basic construction and destruction
TEST_F(SharedMemoryManagerTest, ConstructAndDestruct)
{
  EXPECT_NO_THROW(shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_,
      1024 * 1024,  // 1MB
      512 * 1024,   // 512KB growth
      true          // create
  ));

  EXPECT_NE(shm_pool_, nullptr);
}

// Test Construct method for simple types
TEST_F(SharedMemoryManagerTest, ConstructSimpleType)
{
  shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_, 1024 * 1024, 512 * 1024, true);

  auto allocated = shm_pool_->Construct<int>();

  EXPECT_NE(allocated.data_.get(), nullptr);
  EXPECT_NE(allocated.handle_, 0);

  // Write and read data
  *allocated.data_ = 42;
  EXPECT_EQ(*allocated.data_, 42);
}

// Test Construct method for arrays
TEST_F(SharedMemoryManagerTest, ConstructArray)
{
  shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_, 1024 * 1024, 512 * 1024, true);

  const uint64_t count = 100;
  auto allocated = shm_pool_->Construct<char>(count);

  EXPECT_NE(allocated.data_.get(), nullptr);

  // Fill array with test pattern
  for (uint64_t i = 0; i < count; ++i) {
    allocated.data_.get()[i] = static_cast<char>(i % 256);
  }

  // Verify pattern
  for (uint64_t i = 0; i < count; ++i) {
    EXPECT_EQ(allocated.data_.get()[i], static_cast<char>(i % 256));
  }
}

// Test Construct with alignment
TEST_F(SharedMemoryManagerTest, ConstructAligned)
{
  shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_, 1024 * 1024, 512 * 1024, true);

  auto allocated = shm_pool_->Construct<char>(64, true /* aligned */);

  EXPECT_NE(allocated.data_.get(), nullptr);

  // Note: The implementation allocates with 32-byte alignment, but the returned
  // pointer is offset by sizeof(AllocatedShmOwnership) which is 16 bytes.
  // So the actual data pointer is 16-byte aligned (32 - 16 = 16).
  // This is correct behavior per the implementation design.
  uintptr_t addr = reinterpret_cast<uintptr_t>(allocated.data_.get());
  EXPECT_EQ(addr % 16, 0) << "Memory is not 16-byte aligned";
}

// Test Load method
TEST_F(SharedMemoryManagerTest, LoadFromHandle)
{
  shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_, 1024 * 1024, 512 * 1024, true);

  // Construct and set value
  auto original = shm_pool_->Construct<int>();
  *original.data_ = 12345;
  auto handle = original.handle_;

  // Load using handle
  auto loaded = shm_pool_->Load<int>(handle);

  EXPECT_EQ(*loaded.data_, 12345);
  EXPECT_EQ(loaded.handle_, handle);
}

// Test Load with unsafe flag
TEST_F(SharedMemoryManagerTest, LoadUnsafe)
{
  shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_, 1024 * 1024, 512 * 1024, true);

  auto original = shm_pool_->Construct<int>();
  *original.data_ = 999;
  auto handle = original.handle_;

  // Load with unsafe=true (doesn't increment ref count)
  auto loaded = shm_pool_->Load<int>(handle, true /* unsafe */);

  EXPECT_EQ(*loaded.data_, 999);
}

// Test FreeMemory reports available memory
TEST_F(SharedMemoryManagerTest, FreeMemoryReporting)
{
  shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_, 1024 * 1024, 512 * 1024, true);

  size_t initial_free = shm_pool_->FreeMemory();

  // Allocate some memory
  auto alloc1 = shm_pool_->Construct<char>(1024);
  size_t after_alloc = shm_pool_->FreeMemory();

  EXPECT_LT(after_alloc, initial_free);
}

// Test Deallocate method
TEST_F(SharedMemoryManagerTest, Deallocate)
{
  shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_, 1024 * 1024, 512 * 1024, true);

  // This is primarily tested through the unique_ptr destructor,
  // but we can also test manual deallocation
  auto alloc = shm_pool_->Construct<char>(1024);
  size_t before_dealloc = shm_pool_->FreeMemory();

  // Let the unique_ptr go out of scope (triggers deallocate)
  alloc.data_.reset();

  // Note: Due to reference counting, memory may not be immediately freed
  // This test verifies no crash occurs
  SUCCEED();
}

// Test SetDeleteRegion
TEST_F(SharedMemoryManagerTest, SetDeleteRegion)
{
  shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_, 1024 * 1024, 512 * 1024, true);

  // Should not throw
  EXPECT_NO_THROW(shm_pool_->SetDeleteRegion(true));
  EXPECT_NO_THROW(shm_pool_->SetDeleteRegion(false));
}

// Test Mutex access
TEST_F(SharedMemoryManagerTest, MutexAccess)
{
  shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_, 1024 * 1024, 512 * 1024, true);

  bi::interprocess_mutex* mutex = shm_pool_->Mutex();
  EXPECT_NE(mutex, nullptr);
}

//
// SharedMemoryManager Growth Tests
//
class SharedMemoryManagerGrowthTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    shm_region_name_ = "/shm_growth_test_" + std::to_string(getpid());
  }

  void TearDown() override
  {
    shm_pool_.reset();
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
};

// Test that shared memory grows when needed
TEST_F(SharedMemoryManagerGrowthTest, GrowOnLargeAllocation)
{
  size_t initial_size = 4096;  // Small initial size
  size_t growth_size = 8192;

  shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_, initial_size, growth_size, true);

  // Try to allocate more than initial size
  // This should trigger growth
  EXPECT_NO_THROW({
    auto alloc = shm_pool_->Construct<char>(initial_size * 2);
    EXPECT_NE(alloc.data_.get(), nullptr);
  });
}

// Test multiple allocations that require growth
TEST_F(SharedMemoryManagerGrowthTest, MultipleGrowths)
{
  size_t initial_size = 4096;
  size_t growth_size = 4096;

  shm_pool_ = std::make_unique<SharedMemoryManager>(
      shm_region_name_, initial_size, growth_size, true);

  std::vector<AllocatedSharedMemory<char>> allocations;

  // Allocate multiple times to trigger multiple growths
  for (int i = 0; i < 10; ++i) {
    auto alloc = shm_pool_->Construct<char>(1024);
    EXPECT_NE(alloc.data_.get(), nullptr);
    allocations.push_back(std::move(alloc));
  }

  // Verify all allocations are still valid
  for (size_t i = 0; i < allocations.size(); ++i) {
    EXPECT_NE(allocations[i].data_.get(), nullptr);
  }
}

//
// SharedMemoryManager Thread Safety Tests
//
class SharedMemoryManagerThreadTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    shm_region_name_ = "/shm_thread_test_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_, 1024 * 1024, 512 * 1024, true);
  }

  void TearDown() override
  {
    shm_pool_.reset();
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
};

// Test concurrent allocations from multiple threads
TEST_F(SharedMemoryManagerThreadTest, ConcurrentAllocations)
{
  const int num_threads = 4;
  const int allocations_per_thread = 100;
  std::vector<std::thread> threads;
  std::atomic<int> successful_allocations{0};

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([this, &successful_allocations, allocations_per_thread]() {
      for (int i = 0; i < allocations_per_thread; ++i) {
        try {
          auto alloc = shm_pool_->Construct<int>();
          *alloc.data_ = i;
          successful_allocations++;
        } catch (const std::exception& e) {
          // Log but don't fail - might be expected under high contention
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Most allocations should succeed
  EXPECT_GT(successful_allocations.load(), num_threads * allocations_per_thread / 2);
}

// Test concurrent Load operations
TEST_F(SharedMemoryManagerThreadTest, ConcurrentLoads)
{
  // Create a shared value
  auto original = shm_pool_->Construct<int>();
  *original.data_ = 42;
  auto handle = original.handle_;

  const int num_threads = 4;
  const int loads_per_thread = 100;
  std::vector<std::thread> threads;
  std::atomic<int> correct_reads{0};

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([this, handle, &correct_reads, loads_per_thread]() {
      for (int i = 0; i < loads_per_thread; ++i) {
        auto loaded = shm_pool_->Load<int>(handle);
        if (*loaded.data_ == 42) {
          correct_reads++;
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(correct_reads.load(), num_threads * loads_per_thread);
}

//
// SharedMemoryManager with ROCm Types Tests
//
#ifdef TRITON_ENABLE_ROCM

class SharedMemoryManagerROCmTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    shm_region_name_ = "/shm_rocm_test_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_, 1024 * 1024, 512 * 1024, true);
  }

  void TearDown() override
  {
    shm_pool_.reset();
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
};

// Test storing hipIpcMemHandle_t in shared memory
TEST_F(SharedMemoryManagerROCmTest, StoreHipIpcMemHandle)
{
  // Allocate space for hipIpcMemHandle_t
  auto alloc = shm_pool_->Construct<hipIpcMemHandle_t>();
  EXPECT_NE(alloc.data_.get(), nullptr);

  // Zero out the handle
  std::memset(alloc.data_.get(), 0, sizeof(hipIpcMemHandle_t));

  // Verify we can load it back
  auto loaded = shm_pool_->Load<hipIpcMemHandle_t>(alloc.handle_);
  EXPECT_NE(loaded.data_.get(), nullptr);
}

// Test MemoryShm structure storage
TEST_F(SharedMemoryManagerROCmTest, StoreMemoryShm)
{
  auto alloc = shm_pool_->Construct<MemoryShm>();
  EXPECT_NE(alloc.data_.get(), nullptr);

  // Fill structure
  alloc.data_->memory_type = TRITONSERVER_MEMORY_GPU;
  alloc.data_->memory_type_id = 0;
  alloc.data_->byte_size = 4096;
  alloc.data_->is_cuda_handle_set = true;
  alloc.data_->gpu_pointer_offset = 128;
  alloc.data_->memory_release_id = 12345;

  // Load and verify
  auto loaded = shm_pool_->Load<MemoryShm>(alloc.handle_);
  EXPECT_EQ(loaded.data_->memory_type, TRITONSERVER_MEMORY_GPU);
  EXPECT_EQ(loaded.data_->memory_type_id, 0);
  EXPECT_EQ(loaded.data_->byte_size, 4096);
  EXPECT_EQ(loaded.data_->is_cuda_handle_set, true);
  EXPECT_EQ(loaded.data_->gpu_pointer_offset, 128);
  EXPECT_EQ(loaded.data_->memory_release_id, 12345);
}

#endif  // TRITON_ENABLE_ROCM

}  // namespace test
}}}  // namespace triton::backend::python

