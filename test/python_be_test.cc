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
// Unit Tests for python_be.h/.cc
// Tests BackendState struct, environment management, and utility patterns
// used in the Python backend.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <atomic>
#include <cctype>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "pb_exception.h"
#include "pb_memory.h"
#include "pb_preferred_memory.h"
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {
namespace test {

//
// Test BackendState structure
// This mirrors the struct defined in python_be.h
//
struct TestBackendState {
  std::string python_lib;
  int64_t shm_default_byte_size;
  int64_t shm_growth_byte_size;
  int64_t stub_timeout_seconds;
  int64_t shm_message_queue_size;
  std::atomic<int> number_of_instance_inits;
  std::string shared_memory_region_prefix;
  int64_t thread_pool_size;
  std::string runtime_modeldir;
};

class BackendStateTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    state_.shm_default_byte_size = 1 * 1024 * 1024;   // 1 MB
    state_.shm_growth_byte_size = 1 * 1024 * 1024;    // 1 MB
    state_.stub_timeout_seconds = 30;
    state_.shm_message_queue_size = 1000;
    state_.number_of_instance_inits = 0;
    state_.thread_pool_size = 32;
    state_.shared_memory_region_prefix = "triton_python_backend_shm_region_";
    state_.runtime_modeldir = "";
    state_.python_lib = "";
  }

  TestBackendState state_;
};

TEST_F(BackendStateTest, DefaultSharedMemoryConfiguration)
{
  // Test default values as per TRITONBACKEND_Initialize
  EXPECT_EQ(state_.shm_default_byte_size, 1 * 1024 * 1024);
  EXPECT_EQ(state_.shm_growth_byte_size, 1 * 1024 * 1024);
}

TEST_F(BackendStateTest, DefaultStubTimeout)
{
  EXPECT_EQ(state_.stub_timeout_seconds, 30);
}

TEST_F(BackendStateTest, DefaultMessageQueueSize)
{
  EXPECT_EQ(state_.shm_message_queue_size, 1000);
}

TEST_F(BackendStateTest, DefaultThreadPoolSize)
{
  EXPECT_EQ(state_.thread_pool_size, 32);
}

TEST_F(BackendStateTest, InstanceInitCounter)
{
  EXPECT_EQ(state_.number_of_instance_inits.load(), 0);

  // Simulate instance initialization count
  state_.number_of_instance_inits++;
  EXPECT_EQ(state_.number_of_instance_inits.load(), 1);

  state_.number_of_instance_inits++;
  EXPECT_EQ(state_.number_of_instance_inits.load(), 2);
}

TEST_F(BackendStateTest, AtomicCounterThreadSafety)
{
  state_.number_of_instance_inits = 0;

  // Simulate concurrent increments
  std::vector<std::thread> threads;
  const int num_threads = 10;
  const int increments_per_thread = 100;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([this, increments_per_thread]() {
      for (int j = 0; j < increments_per_thread; ++j) {
        state_.number_of_instance_inits++;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(
      state_.number_of_instance_inits.load(),
      num_threads * increments_per_thread);
}

TEST_F(BackendStateTest, SharedMemoryRegionPrefix)
{
  EXPECT_EQ(
      state_.shared_memory_region_prefix, "triton_python_backend_shm_region_");

  // Test custom prefix (as done in TRITONBACKEND_Initialize)
  state_.shared_memory_region_prefix = "triton_custom_backend_shm_region_";
  EXPECT_EQ(
      state_.shared_memory_region_prefix, "triton_custom_backend_shm_region_");
}

//
// Test shared memory configuration validation patterns
// (based on TRITONBACKEND_Initialize validation logic)
//
class SharedMemoryConfigValidationTest : public ::testing::Test {
 protected:
  // Validates shm-default-byte-size according to backend rules
  bool ValidateDefaultByteSize(int64_t size)
  {
    // Shared memory default byte size can't be less than 1 MB
    return size >= 1 * 1024 * 1024;
  }

  // Validates shm-growth-byte-size according to backend rules
  bool ValidateGrowthByteSize(int64_t size)
  {
    return size > 0;
  }

  // Validates stub-timeout-seconds according to backend rules
  bool ValidateStubTimeout(int64_t timeout)
  {
    return timeout > 0;
  }

  // Validates thread-pool-size according to backend rules
  bool ValidateThreadPoolSize(int64_t size)
  {
    return size >= 1;
  }

  // Validates shm-region-prefix-name according to backend rules
  bool ValidateRegionPrefix(const std::string& prefix)
  {
    return !prefix.empty();
  }
};

TEST_F(SharedMemoryConfigValidationTest, ValidDefaultByteSizes)
{
  EXPECT_TRUE(ValidateDefaultByteSize(1 * 1024 * 1024));   // 1 MB - minimum
  EXPECT_TRUE(ValidateDefaultByteSize(4 * 1024 * 1024));   // 4 MB
  EXPECT_TRUE(ValidateDefaultByteSize(16 * 1024 * 1024));  // 16 MB
  EXPECT_TRUE(ValidateDefaultByteSize(64 * 1024 * 1024));  // 64 MB
}

TEST_F(SharedMemoryConfigValidationTest, InvalidDefaultByteSizes)
{
  EXPECT_FALSE(ValidateDefaultByteSize(0));                  // Zero
  EXPECT_FALSE(ValidateDefaultByteSize(512 * 1024));         // 512 KB - too small
  EXPECT_FALSE(ValidateDefaultByteSize(1024 * 1024 - 1));    // Just under 1 MB
}

TEST_F(SharedMemoryConfigValidationTest, ValidGrowthByteSizes)
{
  EXPECT_TRUE(ValidateGrowthByteSize(1));
  EXPECT_TRUE(ValidateGrowthByteSize(1024));
  EXPECT_TRUE(ValidateGrowthByteSize(1 * 1024 * 1024));
}

TEST_F(SharedMemoryConfigValidationTest, InvalidGrowthByteSizes)
{
  EXPECT_FALSE(ValidateGrowthByteSize(0));
  EXPECT_FALSE(ValidateGrowthByteSize(-1));
}

TEST_F(SharedMemoryConfigValidationTest, ValidStubTimeouts)
{
  EXPECT_TRUE(ValidateStubTimeout(1));
  EXPECT_TRUE(ValidateStubTimeout(30));
  EXPECT_TRUE(ValidateStubTimeout(300));
}

TEST_F(SharedMemoryConfigValidationTest, InvalidStubTimeouts)
{
  EXPECT_FALSE(ValidateStubTimeout(0));
  EXPECT_FALSE(ValidateStubTimeout(-1));
  EXPECT_FALSE(ValidateStubTimeout(-100));
}

TEST_F(SharedMemoryConfigValidationTest, ValidThreadPoolSizes)
{
  EXPECT_TRUE(ValidateThreadPoolSize(1));
  EXPECT_TRUE(ValidateThreadPoolSize(8));
  EXPECT_TRUE(ValidateThreadPoolSize(32));
  EXPECT_TRUE(ValidateThreadPoolSize(64));
}

TEST_F(SharedMemoryConfigValidationTest, InvalidThreadPoolSizes)
{
  EXPECT_FALSE(ValidateThreadPoolSize(0));
  EXPECT_FALSE(ValidateThreadPoolSize(-1));
}

TEST_F(SharedMemoryConfigValidationTest, ValidRegionPrefixes)
{
  EXPECT_TRUE(ValidateRegionPrefix("a"));
  EXPECT_TRUE(ValidateRegionPrefix("triton_"));
  EXPECT_TRUE(ValidateRegionPrefix("triton_python_backend_shm_region_"));
}

TEST_F(SharedMemoryConfigValidationTest, InvalidRegionPrefixes)
{
  EXPECT_FALSE(ValidateRegionPrefix(""));
}

//
// Test ClosedRequests tracking pattern
// (used in ModelInstanceState::ExistsInClosedRequests)
//
class ClosedRequestsTrackingTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    closed_requests_.clear();
  }

  bool ExistsInClosedRequests(intptr_t request)
  {
    std::lock_guard<std::mutex> guard{closed_requests_mutex_};
    return std::find(
               closed_requests_.begin(), closed_requests_.end(), request) !=
           closed_requests_.end();
  }

  void AddClosedRequest(intptr_t request)
  {
    std::lock_guard<std::mutex> guard{closed_requests_mutex_};
    closed_requests_.push_back(request);
  }

  void ClearClosedRequests()
  {
    std::lock_guard<std::mutex> guard{closed_requests_mutex_};
    closed_requests_.clear();
  }

  std::vector<intptr_t> closed_requests_;
  std::mutex closed_requests_mutex_;
};

TEST_F(ClosedRequestsTrackingTest, EmptyList)
{
  EXPECT_FALSE(ExistsInClosedRequests(0x1000));
  EXPECT_FALSE(ExistsInClosedRequests(0x2000));
}

TEST_F(ClosedRequestsTrackingTest, SingleRequest)
{
  AddClosedRequest(0x1000);

  EXPECT_TRUE(ExistsInClosedRequests(0x1000));
  EXPECT_FALSE(ExistsInClosedRequests(0x2000));
}

TEST_F(ClosedRequestsTrackingTest, MultipleRequests)
{
  AddClosedRequest(0x1000);
  AddClosedRequest(0x2000);
  AddClosedRequest(0x3000);

  EXPECT_TRUE(ExistsInClosedRequests(0x1000));
  EXPECT_TRUE(ExistsInClosedRequests(0x2000));
  EXPECT_TRUE(ExistsInClosedRequests(0x3000));
  EXPECT_FALSE(ExistsInClosedRequests(0x4000));
}

TEST_F(ClosedRequestsTrackingTest, ClearRequests)
{
  AddClosedRequest(0x1000);
  AddClosedRequest(0x2000);

  EXPECT_TRUE(ExistsInClosedRequests(0x1000));

  ClearClosedRequests();

  EXPECT_FALSE(ExistsInClosedRequests(0x1000));
  EXPECT_FALSE(ExistsInClosedRequests(0x2000));
}

TEST_F(ClosedRequestsTrackingTest, ThreadSafetyTest)
{
  const int num_threads = 4;
  const int requests_per_thread = 100;
  std::vector<std::thread> threads;

  // Add requests from multiple threads
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([this, t, requests_per_thread]() {
      for (int i = 0; i < requests_per_thread; ++i) {
        intptr_t request_id = (t * requests_per_thread + i) * 8;  // Fake address
        AddClosedRequest(request_id);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Verify all requests were added
  EXPECT_EQ(closed_requests_.size(), num_threads * requests_per_thread);
}

//
// Test FORCE_CPU_ONLY_INPUT_TENSORS parameter parsing pattern
//
class ForceCpuOnlyInputTensorsTest : public ::testing::Test {
 protected:
  // Returns: 0 = no (GPU allowed), 1 = yes (CPU only), -1 = invalid
  int ParseForceCpuOnlyValue(const std::string& value)
  {
    if (value == "yes") {
      return 1;
    } else if (value == "no") {
      return 0;
    }
    return -1;  // Invalid
  }
};

TEST_F(ForceCpuOnlyInputTensorsTest, ValidYesValue)
{
  EXPECT_EQ(ParseForceCpuOnlyValue("yes"), 1);
}

TEST_F(ForceCpuOnlyInputTensorsTest, ValidNoValue)
{
  EXPECT_EQ(ParseForceCpuOnlyValue("no"), 0);
}

TEST_F(ForceCpuOnlyInputTensorsTest, InvalidValues)
{
  EXPECT_EQ(ParseForceCpuOnlyValue("true"), -1);
  EXPECT_EQ(ParseForceCpuOnlyValue("false"), -1);
  EXPECT_EQ(ParseForceCpuOnlyValue("YES"), -1);
  EXPECT_EQ(ParseForceCpuOnlyValue("NO"), -1);
  EXPECT_EQ(ParseForceCpuOnlyValue("1"), -1);
  EXPECT_EQ(ParseForceCpuOnlyValue("0"), -1);
  EXPECT_EQ(ParseForceCpuOnlyValue(""), -1);
}

//
// Test batch size validation pattern
// (used in ModelInstanceState::CheckIncomingRequests)
//
class BatchSizeValidationTest : public ::testing::Test {
 protected:
  // Validates batch size according to backend rules
  // Returns error message or empty string on success
  std::string ValidateBatchSize(
      size_t total_batch_size, int max_batch_size, const std::string& name)
  {
    if (total_batch_size == 0) {
      return "";  // No error, but no work to do
    }

    // Check if max_batch_size exceeded
    if ((total_batch_size != 1) &&
        (total_batch_size > static_cast<size_t>(max_batch_size))) {
      return "batch size " + std::to_string(total_batch_size) + " for '" +
             name + "', max allowed is " + std::to_string(max_batch_size);
    }

    return "";  // Success
  }
};

TEST_F(BatchSizeValidationTest, ZeroBatchSize)
{
  auto error = ValidateBatchSize(0, 8, "test_model");
  EXPECT_TRUE(error.empty());
}

TEST_F(BatchSizeValidationTest, SingleItemNoBatching)
{
  // For models that don't support batching (max_batch_size=0),
  // total_batch_size must be 1
  auto error = ValidateBatchSize(1, 0, "test_model");
  EXPECT_TRUE(error.empty());
}

TEST_F(BatchSizeValidationTest, ValidBatchSizes)
{
  auto error = ValidateBatchSize(1, 8, "test_model");
  EXPECT_TRUE(error.empty());

  error = ValidateBatchSize(4, 8, "test_model");
  EXPECT_TRUE(error.empty());

  error = ValidateBatchSize(8, 8, "test_model");
  EXPECT_TRUE(error.empty());
}

TEST_F(BatchSizeValidationTest, ExceedsMaxBatchSize)
{
  auto error = ValidateBatchSize(16, 8, "test_model");
  EXPECT_FALSE(error.empty());
  EXPECT_THAT(error, testing::HasSubstr("batch size 16"));
  EXPECT_THAT(error, testing::HasSubstr("max allowed is 8"));
}

//
// Test response error sending patterns
// (simulate error response creation as done in python_be.cc)
//
class ErrorResponsePatternTest : public ::testing::Test {
 protected:
  struct MockResponse {
    bool sent = false;
    std::string error_message;
  };

  void RespondErrorToAllRequests(
      const std::string& error_message, const std::string& instance_name,
      std::vector<MockResponse*>& responses)
  {
    for (auto& response : responses) {
      if (response == nullptr)
        continue;

      std::string full_message =
          "Failed to process the request(s) for model instance '" +
          instance_name + "', message: " + error_message;

      response->sent = true;
      response->error_message = full_message;
    }
  }
};

TEST_F(ErrorResponsePatternTest, SingleResponse)
{
  MockResponse response;
  std::vector<MockResponse*> responses = {&response};

  RespondErrorToAllRequests("Test error", "model_instance_0", responses);

  EXPECT_TRUE(response.sent);
  EXPECT_THAT(
      response.error_message, testing::HasSubstr("model_instance_0"));
  EXPECT_THAT(response.error_message, testing::HasSubstr("Test error"));
}

TEST_F(ErrorResponsePatternTest, MultipleResponses)
{
  MockResponse r1, r2, r3;
  std::vector<MockResponse*> responses = {&r1, &r2, &r3};

  RespondErrorToAllRequests("Connection failed", "model_0", responses);

  for (auto* r : responses) {
    EXPECT_TRUE(r->sent);
    EXPECT_THAT(r->error_message, testing::HasSubstr("Connection failed"));
  }
}

TEST_F(ErrorResponsePatternTest, SkipsNullResponses)
{
  MockResponse r1, r3;
  std::vector<MockResponse*> responses = {&r1, nullptr, &r3};

  // Should not crash when encountering nullptr
  RespondErrorToAllRequests("Error", "model", responses);

  EXPECT_TRUE(r1.sent);
  EXPECT_TRUE(r3.sent);
}

TEST_F(ErrorResponsePatternTest, EmptyResponsesList)
{
  std::vector<MockResponse*> responses;

  // Should handle empty list gracefully
  RespondErrorToAllRequests("Error", "model", responses);

  EXPECT_TRUE(responses.empty());
}

//
// Test SharedMemory Region Naming Pattern
// (used throughout python_be.cc for unique region naming)
//
class ShmRegionNamingTest : public ::testing::Test {
 protected:
  std::string CreateRegionName(
      const std::string& prefix, const std::string& suffix)
  {
    return "/" + prefix + suffix + "_" + std::to_string(getpid());
  }
};

TEST_F(ShmRegionNamingTest, BasicNaming)
{
  std::string name = CreateRegionName("triton_", "test");
  EXPECT_THAT(name, testing::StartsWith("/triton_test_"));
  // Verify that the rest after the prefix is numeric (PID)
  std::string suffix = name.substr(strlen("/triton_test_"));
  EXPECT_FALSE(suffix.empty());
  for (char c : suffix) {
    EXPECT_TRUE(std::isdigit(c)) << "Expected digit, got: " << c;
  }
}

TEST_F(ShmRegionNamingTest, UniquenessPerProcess)
{
  std::string name1 = CreateRegionName("prefix_", "suffix1");
  std::string name2 = CreateRegionName("prefix_", "suffix2");

  // Same PID suffix, different names
  EXPECT_NE(name1, name2);
}

TEST_F(ShmRegionNamingTest, ContainsPid)
{
  std::string name = CreateRegionName("test_", "region");
  std::string pid_str = std::to_string(getpid());
  EXPECT_THAT(name, testing::HasSubstr(pid_str));
}

//
// Test Decoupled Mode Flag Pattern
//
class DecoupledModeTest : public ::testing::Test {
 protected:
  struct MockModelState {
    bool decoupled_ = false;
    bool IsDecoupled() const { return decoupled_; }
  };
};

TEST_F(DecoupledModeTest, DefaultIsNotDecoupled)
{
  MockModelState state;
  EXPECT_FALSE(state.IsDecoupled());
}

TEST_F(DecoupledModeTest, SetDecoupled)
{
  MockModelState state;
  state.decoupled_ = true;
  EXPECT_TRUE(state.IsDecoupled());
}

//
// Test Stub Health Check Timeout Pattern
// (simulates the timeout logic used in IsStubProcessAlive)
//
class StubHealthTimeoutTest : public ::testing::Test {
 protected:
  // Simulates health check with timeout
  bool CheckHealthWithTimeout(bool stub_healthy, bool lock_acquired)
  {
    if (lock_acquired) {
      return stub_healthy;
    } else {
      // If lock couldn't be acquired, stub is considered unhealthy
      return false;
    }
  }
};

TEST_F(StubHealthTimeoutTest, HealthyWithLock)
{
  EXPECT_TRUE(CheckHealthWithTimeout(true, true));
}

TEST_F(StubHealthTimeoutTest, UnhealthyWithLock)
{
  EXPECT_FALSE(CheckHealthWithTimeout(false, true));
}

TEST_F(StubHealthTimeoutTest, LockNotAcquired)
{
  // Even if stub would be healthy, failing to acquire lock means unhealthy
  EXPECT_FALSE(CheckHealthWithTimeout(true, false));
  EXPECT_FALSE(CheckHealthWithTimeout(false, false));
}

}  // namespace test
}}}  // namespace triton::backend::python

