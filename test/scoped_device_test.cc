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

#include "pb_utils.h"
#include "pb_exception.h"

namespace triton { namespace backend { namespace python {
namespace test {

//
// ScopedSetDevice Tests
//
class ScopedSetDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Check if HIP is available
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }

    device_count_ = device_count;

    // Initialize device 0 by allocating memory (creates primary context)
    (void)hipSetDevice(0);
    void* ptr = nullptr;
    (void)hipMalloc(&ptr, 1024);
    (void)hipFree(ptr);
  }

  int device_count_ = 0;
};

// Test ScopedSetDevice sets the device correctly
TEST_F(ScopedSetDeviceTest, SetsDeviceCorrectly)
{
  int initial_device;
  ASSERT_EQ(hipGetDevice(&initial_device), hipSuccess);

  {
    ScopedSetDevice scoped(0);
    int current_device;
    ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
    EXPECT_EQ(current_device, 0);
  }
}

// Test ScopedSetDevice restores device on destruction
TEST_F(ScopedSetDeviceTest, RestoresDeviceOnDestruction)
{
  // Ensure we start on device 0
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  int original_device;
  ASSERT_EQ(hipGetDevice(&original_device), hipSuccess);

  {
    // This creates a scoped device change
    ScopedSetDevice scoped(0);
  }

  // After scope ends, should be back to original device
  int restored_device;
  ASSERT_EQ(hipGetDevice(&restored_device), hipSuccess);
  EXPECT_EQ(restored_device, original_device);
}

// Test ScopedSetDevice with same device (no change needed)
TEST_F(ScopedSetDeviceTest, SameDeviceNoChange)
{
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  int original_device;
  ASSERT_EQ(hipGetDevice(&original_device), hipSuccess);

  {
    ScopedSetDevice scoped(0);  // Same device
    int current_device;
    ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
    EXPECT_EQ(current_device, 0);
  }

  int final_device;
  ASSERT_EQ(hipGetDevice(&final_device), hipSuccess);
  EXPECT_EQ(final_device, 0);
}

//
// Multi-Device ScopedSetDevice Tests
//
class ScopedSetDeviceMultiGPUTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count < 2) {
      GTEST_SKIP() << "Multi-GPU tests require at least 2 GPUs";
    }

    device_count_ = device_count;

    // Initialize both devices by creating primary contexts
    for (int i = 0; i < device_count_; ++i) {
      (void)hipSetDevice(i);
      void* ptr = nullptr;
      (void)hipMalloc(&ptr, 1024);
      (void)hipFree(ptr);
    }

    // Return to device 0
    (void)hipSetDevice(0);
  }

  int device_count_ = 0;
};

// Test switching between devices
TEST_F(ScopedSetDeviceMultiGPUTest, SwitchBetweenDevices)
{
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  {
    ScopedSetDevice scoped(1);
    int current_device;
    ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
    EXPECT_EQ(current_device, 1);
  }

  // Should be restored to device 0
  int restored_device;
  ASSERT_EQ(hipGetDevice(&restored_device), hipSuccess);
  EXPECT_EQ(restored_device, 0);
}

// Test nested ScopedSetDevice
TEST_F(ScopedSetDeviceMultiGPUTest, NestedScopedSetDevice)
{
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  {
    ScopedSetDevice outer(1);
    int device_after_outer;
    ASSERT_EQ(hipGetDevice(&device_after_outer), hipSuccess);
    EXPECT_EQ(device_after_outer, 1);

    {
      ScopedSetDevice inner(0);
      int device_after_inner;
      ASSERT_EQ(hipGetDevice(&device_after_inner), hipSuccess);
      EXPECT_EQ(device_after_inner, 0);
    }

    // After inner scope, should be back to device 1
    int device_after_inner_scope;
    ASSERT_EQ(hipGetDevice(&device_after_inner_scope), hipSuccess);
    EXPECT_EQ(device_after_inner_scope, 1);
  }

  // After outer scope, should be back to device 0
  int final_device;
  ASSERT_EQ(hipGetDevice(&final_device), hipSuccess);
  EXPECT_EQ(final_device, 0);
}

// Test ScopedSetDevice in a loop
TEST_F(ScopedSetDeviceMultiGPUTest, ScopedSetDeviceInLoop)
{
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  for (int target_device = 0; target_device < device_count_; ++target_device) {
    {
      ScopedSetDevice scoped(target_device);
      int current_device;
      ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
      EXPECT_EQ(current_device, target_device);
    }

    // Should be back to device 0 after each iteration
    int restored_device;
    ASSERT_EQ(hipGetDevice(&restored_device), hipSuccess);
    EXPECT_EQ(restored_device, 0);
  }
}

// Test ScopedSetDevice with memory operations
TEST_F(ScopedSetDeviceMultiGPUTest, ScopedSetDeviceWithMemoryOps)
{
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  {
    ScopedSetDevice scoped(1);

    // Allocate memory on device 1
    void* ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, 1024);
    EXPECT_EQ(err, hipSuccess);
    EXPECT_NE(ptr, nullptr);

    // Verify memory is on device 1
    hipPointerAttribute_t attrs;
    err = hipPointerGetAttributes(&attrs, ptr);
    EXPECT_EQ(err, hipSuccess);
    EXPECT_EQ(attrs.device, 1);

    (void)hipFree(ptr);
  }

  // Verify we're back on device 0
  int current_device;
  ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
  EXPECT_EQ(current_device, 0);
}

//
// HIPHandler MaybeSetDevice Integration Tests
//
class HIPHandlerMaybeSetDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }

    device_count_ = device_count;

    hip_handler_ = &HIPHandler::getInstance();
    if (!hip_handler_->IsAvailable()) {
      GTEST_SKIP() << "HIP handler not available";
    }

    // Initialize all devices
    for (int i = 0; i < device_count_; ++i) {
      (void)hipSetDevice(i);
      void* ptr = nullptr;
      (void)hipMalloc(&ptr, 1024);
      (void)hipFree(ptr);
    }
    (void)hipSetDevice(0);
  }

  int device_count_ = 0;
  HIPHandler* hip_handler_;
};

// Test MaybeSetDevice only sets device if primary context exists
TEST_F(HIPHandlerMaybeSetDeviceTest, OnlySetsIfPrimaryContextExists)
{
  // Since we initialized all devices in SetUp, MaybeSetDevice should work
  for (int i = 0; i < device_count_; ++i) {
    EXPECT_NO_THROW(hip_handler_->MaybeSetDevice(i));

    int current_device;
    ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
    EXPECT_EQ(current_device, i);
  }
}

// Test ScopedSetDevice destructor uses MaybeSetDevice
TEST_F(HIPHandlerMaybeSetDeviceTest, ScopedSetDeviceUsesHIPHandler)
{
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  {
    // Create a context on device 1 so MaybeSetDevice will work
    ScopedSetDevice scoped(1);

    int current;
    ASSERT_EQ(hipGetDevice(&current), hipSuccess);
    EXPECT_EQ(current, 1);
  }

  // ScopedSetDevice destructor should have called MaybeSetDevice(0)
  int final_device;
  ASSERT_EQ(hipGetDevice(&final_device), hipSuccess);
  EXPECT_EQ(final_device, 0);
}

}  // namespace test
}}}  // namespace triton::backend::python

