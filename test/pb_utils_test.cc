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
// Unit Tests for pb_utils.h/cc
// Tests utility macros, error handling, and shared memory structures.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

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
#include "pb_exception.h"
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {
namespace test {

//
// Test the THROW_IF_HIP_ERROR macro
//
#ifdef TRITON_ENABLE_ROCM
class ThrowIfHipErrorTest : public ::testing::Test {
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

TEST_F(ThrowIfHipErrorTest, SuccessDoesNotThrow)
{
  // hipSuccess should not throw
  EXPECT_NO_THROW({
    THROW_IF_HIP_ERROR(hipSuccess);
  });
}

TEST_F(ThrowIfHipErrorTest, ErrorThrowsWithMessage)
{
  // hipErrorInvalidValue should throw
  EXPECT_THROW({
    THROW_IF_HIP_ERROR(hipErrorInvalidValue);
  }, PythonBackendException);

  // Verify the error message is included
  try {
    THROW_IF_HIP_ERROR(hipErrorInvalidValue);
    FAIL() << "Expected PythonBackendException";
  } catch (const PythonBackendException& e) {
    std::string msg = e.what();
    EXPECT_FALSE(msg.empty());
  }
}

TEST_F(ThrowIfHipErrorTest, DifferentErrorCodes)
{
  // Test various error codes
  EXPECT_THROW({
    THROW_IF_HIP_ERROR(hipErrorOutOfMemory);
  }, PythonBackendException);

  EXPECT_THROW({
    THROW_IF_HIP_ERROR(hipErrorNotInitialized);
  }, PythonBackendException);
}
#endif  // TRITON_ENABLE_ROCM

//
// Test PythonBackendException
//
class PythonBackendExceptionTest : public ::testing::Test {};

TEST_F(PythonBackendExceptionTest, BasicConstruction)
{
  PythonBackendException ex("Test error message");
  EXPECT_STREQ(ex.what(), "Test error message");
}

TEST_F(PythonBackendExceptionTest, EmptyMessage)
{
  PythonBackendException ex("");
  EXPECT_STREQ(ex.what(), "");
}

TEST_F(PythonBackendExceptionTest, LongMessage)
{
  std::string long_msg(1000, 'X');
  PythonBackendException ex(long_msg);
  EXPECT_EQ(std::string(ex.what()), long_msg);
}

TEST_F(PythonBackendExceptionTest, ThrowAndCatch)
{
  const char* error_msg = "Custom error";
  bool caught = false;

  try {
    throw PythonBackendException(error_msg);
  } catch (const PythonBackendException& e) {
    caught = true;
    EXPECT_STREQ(e.what(), error_msg);
  }

  EXPECT_TRUE(caught);
}

//
// Test shared memory data structures layout
//
class SharedMemoryStructuresTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    shm_region_name_ = "/pb_utils_test_" + std::to_string(getpid());
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

TEST_F(SharedMemoryStructuresTest, ResponseBatchStructure)
{
  auto alloc = shm_pool_->Construct<ResponseBatch>();
  ResponseBatch* rb = alloc.data_.get();

  // Initialize and verify fields
  rb->batch_size = 5;
  rb->has_error = false;
  rb->is_error_set = false;
  rb->cleanup = true;
  rb->response_size = 10;

  EXPECT_EQ(rb->batch_size, 5);
  EXPECT_FALSE(rb->has_error);
  EXPECT_FALSE(rb->is_error_set);
  EXPECT_TRUE(rb->cleanup);
  EXPECT_EQ(rb->response_size, 10);
}

TEST_F(SharedMemoryStructuresTest, ResponseBatchWithError)
{
  auto alloc = shm_pool_->Construct<ResponseBatch>();
  ResponseBatch* rb = alloc.data_.get();

  rb->has_error = true;
  rb->is_error_set = true;
  rb->batch_size = 1;

  EXPECT_TRUE(rb->has_error);
  EXPECT_TRUE(rb->is_error_set);
}

TEST_F(SharedMemoryStructuresTest, RequestBatchStructure)
{
  auto alloc = shm_pool_->Construct<RequestBatch>();
  RequestBatch* req = alloc.data_.get();

  req->batch_size = 8;
  EXPECT_EQ(req->batch_size, 8);
}

TEST_F(SharedMemoryStructuresTest, IPCControlShmStructure)
{
  auto alloc = shm_pool_->Construct<IPCControlShm>();
  IPCControlShm* ipc = alloc.data_.get();

  ipc->stub_health = true;
  ipc->parent_health = true;
  ipc->uses_env = false;
  ipc->decoupled = true;

  EXPECT_TRUE(ipc->stub_health);
  EXPECT_TRUE(ipc->parent_health);
  EXPECT_FALSE(ipc->uses_env);
  EXPECT_TRUE(ipc->decoupled);
}

TEST_F(SharedMemoryStructuresTest, LogSendMessageStructure)
{
  auto alloc = shm_pool_->Construct<LogSendMessage>();
  LogSendMessage* log = alloc.data_.get();

  log->line = 42;
  log->level = LogLevel::ERROR;
  log->waiting_on_stub = false;

  EXPECT_EQ(log->line, 42);
  EXPECT_EQ(log->level, LogLevel::ERROR);
  EXPECT_FALSE(log->waiting_on_stub);
}

TEST_F(SharedMemoryStructuresTest, LogLevelEnum)
{
  EXPECT_EQ(static_cast<int>(LogLevel::INFO), 0);
  EXPECT_EQ(static_cast<int>(LogLevel::WARNING), 1);
  EXPECT_EQ(static_cast<int>(LogLevel::ERROR), 2);
  EXPECT_EQ(static_cast<int>(LogLevel::VERBOSE), 3);
}

TEST_F(SharedMemoryStructuresTest, MetricKindEnum)
{
  EXPECT_EQ(static_cast<int>(MetricKind::COUNTER), 0);
  EXPECT_EQ(static_cast<int>(MetricKind::GAUGE), 1);
}

TEST_F(SharedMemoryStructuresTest, IsCancelledMessageStructure)
{
  auto alloc = shm_pool_->Construct<IsCancelledMessage>();
  IsCancelledMessage* msg = alloc.data_.get();

  msg->response_factory_address = 0x12345678;
  msg->request_address = 0x87654321;
  msg->is_cancelled = true;

  EXPECT_EQ(msg->response_factory_address, 0x12345678);
  EXPECT_EQ(msg->request_address, 0x87654321);
  EXPECT_TRUE(msg->is_cancelled);
}

TEST_F(SharedMemoryStructuresTest, CustomMetricsMessageStructure)
{
  auto alloc = shm_pool_->Construct<CustomMetricsMessage>();
  CustomMetricsMessage* msg = alloc.data_.get();

  msg->has_error = false;
  msg->is_error_set = false;
  msg->value = 3.14159;
  msg->address = reinterpret_cast<void*>(0xDEADBEEF);

  EXPECT_FALSE(msg->has_error);
  EXPECT_DOUBLE_EQ(msg->value, 3.14159);
  EXPECT_EQ(msg->address, reinterpret_cast<void*>(0xDEADBEEF));
}

TEST_F(SharedMemoryStructuresTest, ResponseSendMessageStructure)
{
  auto alloc = shm_pool_->Construct<ResponseSendMessage>();
  ResponseSendMessage* msg = alloc.data_.get();

  msg->is_stub_turn = true;
  msg->has_error = false;
  msg->is_error_set = false;
  msg->request_address = 12345;
  msg->response_factory_address = 67890;
  msg->flags = 0x1;

  EXPECT_TRUE(msg->is_stub_turn);
  EXPECT_FALSE(msg->has_error);
  EXPECT_EQ(msg->request_address, 12345);
  EXPECT_EQ(msg->response_factory_address, 67890);
  EXPECT_EQ(msg->flags, 0x1);
}


//
// Test interprocess mutex and condition variable initialization
//
class IPCPrimitivesTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    shm_region_name_ = "/ipc_primitives_test_" + std::to_string(getpid());
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

TEST_F(IPCPrimitivesTest, MutexLockUnlock)
{
  auto alloc = shm_pool_->Construct<SendMessageBase>();
  SendMessageBase* msg = alloc.data_.get();

  // Test basic lock/unlock
  msg->mu.lock();
  msg->waiting_on_stub = true;
  msg->mu.unlock();

  EXPECT_TRUE(msg->waiting_on_stub);
}

TEST_F(IPCPrimitivesTest, ScopedLock)
{
  auto alloc = shm_pool_->Construct<SendMessageBase>();
  SendMessageBase* msg = alloc.data_.get();
  msg->waiting_on_stub = false;

  {
    bi::scoped_lock<bi::interprocess_mutex> lock(msg->mu);
    msg->waiting_on_stub = true;
  }

  EXPECT_TRUE(msg->waiting_on_stub);
}

//
// Test RequestBatch with handles array pattern (used in python_be.cc)
//
TEST_F(SharedMemoryStructuresTest, RequestBatchWithHandles)
{
  const uint32_t request_count = 4;
  size_t total_size = sizeof(RequestBatch) +
                      request_count * sizeof(bi::managed_external_buffer::handle_t);

  auto alloc = shm_pool_->Construct<char>(total_size);
  char* data = alloc.data_.get();

  RequestBatch* req_batch = reinterpret_cast<RequestBatch*>(data);
  req_batch->batch_size = request_count;

  bi::managed_external_buffer::handle_t* handles =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          data + sizeof(RequestBatch));

  // Set handles
  for (uint32_t i = 0; i < request_count; ++i) {
    handles[i] = i * 100;
  }

  // Verify
  EXPECT_EQ(req_batch->batch_size, request_count);
  for (uint32_t i = 0; i < request_count; ++i) {
    EXPECT_EQ(handles[i], i * 100);
  }
}

//
// Test ResponseBatch with handles array pattern (used in python_be.cc)
//
TEST_F(SharedMemoryStructuresTest, ResponseBatchWithHandles)
{
  const uint32_t response_count = 3;
  size_t total_size = sizeof(ResponseBatch) +
                      response_count * sizeof(bi::managed_external_buffer::handle_t);

  auto alloc = shm_pool_->Construct<char>(total_size);
  char* data = alloc.data_.get();

  ResponseBatch* resp_batch = reinterpret_cast<ResponseBatch*>(data);
  resp_batch->batch_size = response_count;
  resp_batch->has_error = false;
  resp_batch->is_error_set = false;
  resp_batch->cleanup = false;
  resp_batch->response_size = response_count;

  bi::managed_external_buffer::handle_t* handles =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          data + sizeof(ResponseBatch));

  // Set handles
  for (uint32_t i = 0; i < response_count; ++i) {
    handles[i] = (i + 1) * 1000;
  }

  // Verify
  EXPECT_EQ(resp_batch->batch_size, response_count);
  EXPECT_EQ(resp_batch->response_size, response_count);
  for (uint32_t i = 0; i < response_count; ++i) {
    EXPECT_EQ(handles[i], (i + 1) * 1000);
  }
}

}  // namespace test
}}}  // namespace triton::backend::python

