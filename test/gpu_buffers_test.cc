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

#include <memory>
#include <string>

#include "gpu_buffers.h"
#include "pb_exception.h"
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {
namespace test {

//
// GPUBuffersHelper Tests
//
class GPUBuffersHelperTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    shm_region_name_ = "/gpu_buffers_test_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_,
        1024 * 1024,
        512 * 1024,
        true);
  }

  void TearDown() override
  {
    shm_pool_.reset();
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
};

// Test basic construction
TEST_F(GPUBuffersHelperTest, Construction)
{
  GPUBuffersHelper helper;
  // Should not throw
  SUCCEED();
}

// Test adding buffers
TEST_F(GPUBuffersHelperTest, AddBuffer)
{
  GPUBuffersHelper helper;

  // Create some dummy handles
  auto alloc1 = shm_pool_->Construct<int>();
  auto alloc2 = shm_pool_->Construct<int>();

  EXPECT_NO_THROW(helper.AddBuffer(alloc1.handle_));
  EXPECT_NO_THROW(helper.AddBuffer(alloc2.handle_));
}

// Test Complete
TEST_F(GPUBuffersHelperTest, Complete)
{
  GPUBuffersHelper helper;

  auto alloc_1 = shm_pool_->Construct<int>();
  auto alloc_2 = shm_pool_->Construct<int>();
  *alloc_1.data_ = 1;
  *alloc_2.data_ = 2;
  helper.AddBuffer(alloc_1.handle_);
  helper.AddBuffer(alloc_2.handle_);

  EXPECT_NO_THROW(helper.Complete(shm_pool_));

  auto handle = helper.ShmHandle();
  EXPECT_NE(handle, 0);
  auto loaded = shm_pool_->Load<GPUBuffersShm>(handle);
  EXPECT_EQ(loaded.data_->buffer_count, 2);

  auto buffer_handles = shm_pool_->Load<bi::managed_external_buffer::handle_t>(
      loaded.data_->buffers);
  ASSERT_TRUE(loaded.data_->success);
  ASSERT_EQ(buffer_handles.data_.get()[0], alloc_1.handle_);
  ASSERT_EQ(buffer_handles.data_.get()[1], alloc_2.handle_);

  auto loaded_1 = shm_pool_->Load<int>(buffer_handles.data_.get()[0]);
  auto loaded_2 = shm_pool_->Load<int>(buffer_handles.data_.get()[1]);
  ASSERT_EQ(*loaded_1.data_, 1);
  ASSERT_EQ(*loaded_2.data_, 2);
}

// Test Complete can only be called once
TEST_F(GPUBuffersHelperTest, CompleteCalledTwiceThrows)
{
  GPUBuffersHelper helper;

  auto alloc = shm_pool_->Construct<int>();
  helper.AddBuffer(alloc.handle_);

  helper.Complete(shm_pool_);

  EXPECT_THROW(helper.Complete(shm_pool_), PythonBackendException);
}

// Test AddBuffer after Complete throws
TEST_F(GPUBuffersHelperTest, AddBufferAfterCompleteThrows)
{
  GPUBuffersHelper helper;

  auto alloc = shm_pool_->Construct<int>();
  helper.AddBuffer(alloc.handle_);
  helper.Complete(shm_pool_);

  auto alloc2 = shm_pool_->Construct<int>();
  EXPECT_THROW(helper.AddBuffer(alloc2.handle_), PythonBackendException);
}

// Test SetError
TEST_F(GPUBuffersHelperTest, SetError)
{
  GPUBuffersHelper helper;

  helper.SetError(shm_pool_, "Test error message");
  helper.Complete(shm_pool_);

  auto handle = helper.ShmHandle();
  EXPECT_NE(handle, 0);

  // Load and verify error state
  auto loaded = shm_pool_->Load<GPUBuffersShm>(handle);
  EXPECT_FALSE(loaded.data_->success);
}

// Test empty buffers
TEST_F(GPUBuffersHelperTest, EmptyBuffers)
{
  GPUBuffersHelper helper;

  helper.Complete(shm_pool_);

  auto handle = helper.ShmHandle();
  auto loaded = shm_pool_->Load<GPUBuffersShm>(handle);

  EXPECT_TRUE(loaded.data_->success);
  EXPECT_EQ(loaded.data_->buffer_count, 0);
}

//
// Error Macro Tests
//
class ErrorMacroTest : public ::testing::Test {
};

// Test THROW_IF_ERROR macro
TEST_F(ErrorMacroTest, ThrowIfErrorSuccess)
{
  EXPECT_NO_THROW({
    THROW_IF_ERROR("Should not throw", 0);
  });
}

TEST_F(ErrorMacroTest, ThrowIfErrorFailure)
{
  EXPECT_THROW({
    THROW_IF_ERROR("Expected failure", -1);
  }, PythonBackendException);
}

#ifdef TRITON_ENABLE_ROCM

// Test THROW_IF_HIP_ERROR macro
class HIPErrorMacroTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }
  }
};

TEST_F(HIPErrorMacroTest, ThrowIfHipErrorSuccess)
{
  EXPECT_NO_THROW({
    THROW_IF_HIP_ERROR(hipSuccess);
  });
}

TEST_F(HIPErrorMacroTest, ThrowIfHipErrorFailure)
{
  EXPECT_THROW({
    THROW_IF_HIP_ERROR(hipErrorInvalidValue);
  }, PythonBackendException);
}

TEST_F(HIPErrorMacroTest, HipErrorMessageContainsDescription)
{
  try {
    THROW_IF_HIP_ERROR(hipErrorInvalidValue);
    FAIL() << "Expected exception not thrown";
  } catch (const PythonBackendException& e) {
    std::string msg = e.what();
    // Should contain some description of the error
    EXPECT_FALSE(msg.empty());
  }
}

#endif  // TRITON_ENABLE_ROCM

}  // namespace test
}}}  // namespace triton::backend::python

