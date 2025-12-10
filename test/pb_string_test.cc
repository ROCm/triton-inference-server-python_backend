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
// Unit Tests for pb_string.h/.cc
// Tests PbString class for string storage in shared memory
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <memory>
#include <string>

#include "pb_string.h"
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {
namespace test {

//
// Test fixture with SharedMemoryManager setup
//
class PbStringTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Create a unique shared memory region for each test
    shm_region_name_ =
        "/pb_string_test_" + std::to_string(getpid()) + "_" +
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
// Basic Creation Tests
//
TEST_F(PbStringTest, CreateEmptyString)
{
  std::string empty_str = "";
  auto pb_string = PbString::Create(shm_pool_, empty_str);

  ASSERT_NE(pb_string, nullptr);
  EXPECT_EQ(pb_string->String(), empty_str);
  EXPECT_EQ(pb_string->Size(), sizeof(StringShm) + 0);
}

TEST_F(PbStringTest, CreateSimpleString)
{
  std::string test_str = "Hello, World!";
  auto pb_string = PbString::Create(shm_pool_, test_str);

  ASSERT_NE(pb_string, nullptr);
  EXPECT_EQ(pb_string->String(), test_str);
}

TEST_F(PbStringTest, CreateLongString)
{
  // Create a string with 10000 characters
  std::string long_str(10000, 'x');
  auto pb_string = PbString::Create(shm_pool_, long_str);

  ASSERT_NE(pb_string, nullptr);
  EXPECT_EQ(pb_string->String(), long_str);
  EXPECT_EQ(pb_string->String().length(), 10000u);
}

TEST_F(PbStringTest, CreateStringWithSpecialCharacters)
{
  std::string special_str = "Hello\nWorld\t!\0Test";
  auto pb_string = PbString::Create(shm_pool_, special_str);

  ASSERT_NE(pb_string, nullptr);
  EXPECT_EQ(pb_string->String(), special_str);
}

TEST_F(PbStringTest, CreateStringWithUnicode)
{
  std::string unicode_str = "Hello 世界 🌍";
  auto pb_string = PbString::Create(shm_pool_, unicode_str);

  ASSERT_NE(pb_string, nullptr);
  EXPECT_EQ(pb_string->String(), unicode_str);
}

//
// Load from Shared Memory Tests
//
TEST_F(PbStringTest, SaveAndLoadFromSharedMemory)
{
  std::string test_str = "Test string for save/load";
  auto pb_string = PbString::Create(shm_pool_, test_str);
  ASSERT_NE(pb_string, nullptr);

  auto handle = pb_string->ShmHandle();
  EXPECT_NE(handle, 0u);

  // Load from shared memory using the handle
  auto loaded = PbString::LoadFromSharedMemory(shm_pool_, handle);
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->String(), test_str);
}

TEST_F(PbStringTest, LoadEmptyString)
{
  std::string empty_str = "";
  auto pb_string = PbString::Create(shm_pool_, empty_str);
  auto handle = pb_string->ShmHandle();

  auto loaded = PbString::LoadFromSharedMemory(shm_pool_, handle);
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->String(), empty_str);
}

TEST_F(PbStringTest, LoadLongString)
{
  std::string long_str(5000, 'a');
  auto pb_string = PbString::Create(shm_pool_, long_str);
  auto handle = pb_string->ShmHandle();

  auto loaded = PbString::LoadFromSharedMemory(shm_pool_, handle);
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->String(), long_str);
}

//
// ShmStructSize Tests
//
TEST_F(PbStringTest, ShmStructSizeEmpty)
{
  std::string empty_str = "";
  size_t expected_size = sizeof(StringShm) + empty_str.size();
  EXPECT_EQ(PbString::ShmStructSize(empty_str), expected_size);
}

TEST_F(PbStringTest, ShmStructSizeSimple)
{
  std::string test_str = "Hello";
  size_t expected_size = sizeof(StringShm) + test_str.size();
  EXPECT_EQ(PbString::ShmStructSize(test_str), expected_size);
}

TEST_F(PbStringTest, ShmStructSizeLong)
{
  std::string long_str(1000, 'x');
  size_t expected_size = sizeof(StringShm) + long_str.size();
  EXPECT_EQ(PbString::ShmStructSize(long_str), expected_size);
}

//
// Size() method Tests
//
TEST_F(PbStringTest, SizeMatchesShmStructSize)
{
  std::string test_str = "Test string";
  auto pb_string = PbString::Create(shm_pool_, test_str);

  EXPECT_EQ(pb_string->Size(), PbString::ShmStructSize(test_str));
}

//
// MutableString Tests
//
TEST_F(PbStringTest, MutableStringAccess)
{
  std::string test_str = "Hello";
  auto pb_string = PbString::Create(shm_pool_, test_str);

  char* mutable_ptr = pb_string->MutableString();
  ASSERT_NE(mutable_ptr, nullptr);

  // Verify content
  EXPECT_EQ(std::string(mutable_ptr, test_str.size()), test_str);
}

TEST_F(PbStringTest, MutableStringModification)
{
  std::string test_str = "Hello";
  auto pb_string = PbString::Create(shm_pool_, test_str);

  char* mutable_ptr = pb_string->MutableString();
  ASSERT_NE(mutable_ptr, nullptr);

  // Modify the string in place
  mutable_ptr[0] = 'J';

  EXPECT_EQ(pb_string->String(), "Jello");
}

//
// ShmHandle Tests
//
TEST_F(PbStringTest, ShmHandleIsNonZero)
{
  std::string test_str = "Test";
  auto pb_string = PbString::Create(shm_pool_, test_str);

  EXPECT_NE(pb_string->ShmHandle(), 0u);
}

TEST_F(PbStringTest, ShmHandleIsDifferentForDifferentStrings)
{
  auto pb_string1 = PbString::Create(shm_pool_, "First");
  auto pb_string2 = PbString::Create(shm_pool_, "Second");

  EXPECT_NE(pb_string1->ShmHandle(), pb_string2->ShmHandle());
}

//
// Multiple Strings Tests
//
TEST_F(PbStringTest, CreateMultipleStrings)
{
  std::vector<std::unique_ptr<PbString>> strings;
  std::vector<std::string> test_values = {
      "First", "Second", "Third", "Fourth", "Fifth"};

  for (const auto& val : test_values) {
    strings.push_back(PbString::Create(shm_pool_, val));
  }

  for (size_t i = 0; i < test_values.size(); ++i) {
    EXPECT_EQ(strings[i]->String(), test_values[i]);
  }
}

TEST_F(PbStringTest, LoadMultipleStringsFromHandles)
{
  std::vector<bi::managed_external_buffer::handle_t> handles;
  std::vector<std::unique_ptr<PbString>> pb_strings;  // Keep alive
  std::vector<std::string> test_values = {"One", "Two", "Three"};

  // Create and store handles (keep PbString objects alive)
  for (const auto& val : test_values) {
    auto pb_string = PbString::Create(shm_pool_, val);
    handles.push_back(pb_string->ShmHandle());
    pb_strings.push_back(std::move(pb_string));
  }

  // Load all strings from handles
  for (size_t i = 0; i < test_values.size(); ++i) {
    auto loaded = PbString::LoadFromSharedMemory(shm_pool_, handles[i]);
    EXPECT_EQ(loaded->String(), test_values[i]);
  }
}

//
// Edge Cases
//
TEST_F(PbStringTest, StringWithNullBytes)
{
  // String containing embedded null bytes
  std::string null_str = std::string("Hello\0World", 11);
  auto pb_string = PbString::Create(shm_pool_, null_str);

  ASSERT_NE(pb_string, nullptr);
  EXPECT_EQ(pb_string->String().length(), 11u);
  EXPECT_EQ(pb_string->String(), null_str);
}

TEST_F(PbStringTest, BinaryData)
{
  // Binary data
  std::string binary_data;
  for (int i = 0; i < 256; ++i) {
    binary_data += static_cast<char>(i);
  }

  auto pb_string = PbString::Create(shm_pool_, binary_data);

  ASSERT_NE(pb_string, nullptr);
  EXPECT_EQ(pb_string->String().length(), 256u);
  EXPECT_EQ(pb_string->String(), binary_data);
}

TEST_F(PbStringTest, VeryLongString)
{
  // 100KB string
  std::string very_long(100 * 1024, 'z');
  auto pb_string = PbString::Create(shm_pool_, very_long);

  ASSERT_NE(pb_string, nullptr);
  EXPECT_EQ(pb_string->String(), very_long);
}

}  // namespace test
}}}  // namespace triton::backend::python

