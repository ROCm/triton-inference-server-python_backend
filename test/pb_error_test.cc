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
// Unit Tests for pb_error.h/.cc
// Tests PbError class for error handling with shared memory
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <memory>
#include <string>

#include "pb_error.h"
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {
namespace test {

//
// Test fixture with SharedMemoryManager setup
//
class PbErrorTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Create a unique shared memory region for each test
    shm_region_name_ =
        "/pb_error_test_" + std::to_string(getpid()) + "_" +
        std::to_string(reinterpret_cast<uintptr_t>(this));

    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_, 1024 * 1024 /* 1MB */, 1024 * 1024 /* growth */,
        true /* create */);
  }

  void TearDown() override { shm_pool_.reset(); }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
};

//
// Basic Construction Tests
//
TEST_F(PbErrorTest, ConstructWithMessageOnly)
{
  std::string message = "Test error message";
  PbError error(message);

  EXPECT_EQ(error.Message(), message);
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_INTERNAL);  // Default code
}

TEST_F(PbErrorTest, ConstructWithMessageAndCode)
{
  std::string message = "Not found error";
  PbError error(message, TRITONSERVER_ERROR_NOT_FOUND);

  EXPECT_EQ(error.Message(), message);
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_NOT_FOUND);
}

TEST_F(PbErrorTest, ConstructWithEmptyMessage)
{
  std::string message = "";
  PbError error(message);

  EXPECT_EQ(error.Message(), message);
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_INTERNAL);
}

//
// Different Error Codes Tests
//
TEST_F(PbErrorTest, ErrorCodeUnknown)
{
  PbError error("Unknown error", TRITONSERVER_ERROR_UNKNOWN);
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_UNKNOWN);
}

TEST_F(PbErrorTest, ErrorCodeInternal)
{
  PbError error("Internal error", TRITONSERVER_ERROR_INTERNAL);
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_INTERNAL);
}

TEST_F(PbErrorTest, ErrorCodeNotFound)
{
  PbError error("Not found", TRITONSERVER_ERROR_NOT_FOUND);
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_NOT_FOUND);
}

TEST_F(PbErrorTest, ErrorCodeInvalidArg)
{
  PbError error("Invalid argument", TRITONSERVER_ERROR_INVALID_ARG);
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_INVALID_ARG);
}

TEST_F(PbErrorTest, ErrorCodeUnavailable)
{
  PbError error("Unavailable", TRITONSERVER_ERROR_UNAVAILABLE);
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_UNAVAILABLE);
}

TEST_F(PbErrorTest, ErrorCodeUnsupported)
{
  PbError error("Unsupported", TRITONSERVER_ERROR_UNSUPPORTED);
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_UNSUPPORTED);
}

TEST_F(PbErrorTest, ErrorCodeAlreadyExists)
{
  PbError error("Already exists", TRITONSERVER_ERROR_ALREADY_EXISTS);
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_ALREADY_EXISTS);
}

//
// Save to Shared Memory Tests
//
TEST_F(PbErrorTest, SaveToSharedMemory)
{
  std::string message = "Error to save";
  PbError error(message, TRITONSERVER_ERROR_NOT_FOUND);

  error.SaveToSharedMemory(shm_pool_);

  auto handle = error.ShmHandle();
  EXPECT_NE(handle, 0u);
}

TEST_F(PbErrorTest, SaveEmptyMessageToSharedMemory)
{
  std::string message = "";
  PbError error(message, TRITONSERVER_ERROR_INTERNAL);

  error.SaveToSharedMemory(shm_pool_);

  auto handle = error.ShmHandle();
  EXPECT_NE(handle, 0u);
}

TEST_F(PbErrorTest, SaveLongMessageToSharedMemory)
{
  std::string message(5000, 'e');  // 5000 character message
  PbError error(message, TRITONSERVER_ERROR_INTERNAL);

  error.SaveToSharedMemory(shm_pool_);

  auto handle = error.ShmHandle();
  EXPECT_NE(handle, 0u);
}

//
// Load from Shared Memory Tests
//
TEST_F(PbErrorTest, SaveAndLoadFromSharedMemory)
{
  std::string message = "Roundtrip error message";
  TRITONSERVER_Error_Code code = TRITONSERVER_ERROR_INVALID_ARG;

  PbError original(message, code);
  original.SaveToSharedMemory(shm_pool_);

  auto handle = original.ShmHandle();
  auto loaded = PbError::LoadFromSharedMemory(shm_pool_, handle);

  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->Message(), message);
  EXPECT_EQ(loaded->Code(), code);
}

TEST_F(PbErrorTest, LoadPreservesAllErrorCodes)
{
  std::vector<TRITONSERVER_Error_Code> codes = {
      TRITONSERVER_ERROR_UNKNOWN,
      TRITONSERVER_ERROR_INTERNAL,
      TRITONSERVER_ERROR_NOT_FOUND,
      TRITONSERVER_ERROR_INVALID_ARG,
      TRITONSERVER_ERROR_UNAVAILABLE,
      TRITONSERVER_ERROR_UNSUPPORTED,
      TRITONSERVER_ERROR_ALREADY_EXISTS,
  };

  for (auto code : codes) {
    PbError error("Test message", code);
    error.SaveToSharedMemory(shm_pool_);

    auto loaded = PbError::LoadFromSharedMemory(shm_pool_, error.ShmHandle());

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->Code(), code)
        << "Code mismatch for error code " << static_cast<int>(code);
  }
}

TEST_F(PbErrorTest, LoadEmptyMessage)
{
  std::string message = "";
  PbError error(message, TRITONSERVER_ERROR_INTERNAL);
  error.SaveToSharedMemory(shm_pool_);

  auto loaded = PbError::LoadFromSharedMemory(shm_pool_, error.ShmHandle());

  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->Message(), message);
}

TEST_F(PbErrorTest, LoadLongMessage)
{
  std::string message(10000, 'x');
  PbError error(message, TRITONSERVER_ERROR_INTERNAL);
  error.SaveToSharedMemory(shm_pool_);

  auto loaded = PbError::LoadFromSharedMemory(shm_pool_, error.ShmHandle());

  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->Message(), message);
}

//
// Multiple Errors Tests
//
TEST_F(PbErrorTest, CreateMultipleErrors)
{
  std::vector<std::pair<std::string, TRITONSERVER_Error_Code>> error_data = {
      {"First error", TRITONSERVER_ERROR_UNKNOWN},
      {"Second error", TRITONSERVER_ERROR_NOT_FOUND},
      {"Third error", TRITONSERVER_ERROR_INVALID_ARG},
  };

  std::vector<bi::managed_external_buffer::handle_t> handles;
  std::vector<std::unique_ptr<PbError>> errors;  // Keep alive

  // Save all errors (keep PbError objects alive)
  for (const auto& data : error_data) {
    auto error = std::make_unique<PbError>(data.first, data.second);
    error->SaveToSharedMemory(shm_pool_);
    handles.push_back(error->ShmHandle());
    errors.push_back(std::move(error));
  }

  // Load and verify all errors
  for (size_t i = 0; i < error_data.size(); ++i) {
    auto loaded = PbError::LoadFromSharedMemory(shm_pool_, handles[i]);
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->Message(), error_data[i].first);
    EXPECT_EQ(loaded->Code(), error_data[i].second);
  }
}

//
// Edge Cases
//
TEST_F(PbErrorTest, MessageWithSpecialCharacters)
{
  std::string message = "Error: failed to process\n\tDetails: \"test\"\0end";
  PbError error(message, TRITONSERVER_ERROR_INTERNAL);
  error.SaveToSharedMemory(shm_pool_);

  auto loaded = PbError::LoadFromSharedMemory(shm_pool_, error.ShmHandle());

  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->Message(), message);
}

TEST_F(PbErrorTest, MessageWithUnicode)
{
  std::string message = "Error: 失败 🔥 エラー";
  PbError error(message, TRITONSERVER_ERROR_INTERNAL);
  error.SaveToSharedMemory(shm_pool_);

  auto loaded = PbError::LoadFromSharedMemory(shm_pool_, error.ShmHandle());

  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->Message(), message);
}

TEST_F(PbErrorTest, ShmHandleUniquePerError)
{
  PbError error1("Error 1", TRITONSERVER_ERROR_INTERNAL);
  PbError error2("Error 2", TRITONSERVER_ERROR_INTERNAL);

  error1.SaveToSharedMemory(shm_pool_);
  error2.SaveToSharedMemory(shm_pool_);

  EXPECT_NE(error1.ShmHandle(), error2.ShmHandle());
}

//
// Message Reference Tests
//
TEST_F(PbErrorTest, MessageReturnsByConstReference)
{
  std::string message = "Test message";
  PbError error(message, TRITONSERVER_ERROR_INTERNAL);

  const std::string& ref1 = error.Message();
  const std::string& ref2 = error.Message();

  // Both references should point to the same string
  EXPECT_EQ(&ref1, &ref2);
  EXPECT_EQ(ref1, message);
}

//
// Realistic Error Scenarios
//
TEST_F(PbErrorTest, ModelNotFoundError)
{
  PbError error(
      "Model 'my_model' version '1' is not found",
      TRITONSERVER_ERROR_NOT_FOUND);

  EXPECT_THAT(error.Message(), testing::HasSubstr("my_model"));
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_NOT_FOUND);
}

TEST_F(PbErrorTest, InvalidInputError)
{
  PbError error(
      "Invalid input: expected shape [1,3,224,224], got [1,3,256,256]",
      TRITONSERVER_ERROR_INVALID_ARG);

  EXPECT_THAT(error.Message(), testing::HasSubstr("Invalid input"));
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_INVALID_ARG);
}

TEST_F(PbErrorTest, InternalProcessingError)
{
  PbError error(
      "Internal error: Python exception in model execute()",
      TRITONSERVER_ERROR_INTERNAL);

  EXPECT_THAT(error.Message(), testing::HasSubstr("Python exception"));
  EXPECT_EQ(error.Code(), TRITONSERVER_ERROR_INTERNAL);
}

}  // namespace test
}}}  // namespace triton::backend::python

