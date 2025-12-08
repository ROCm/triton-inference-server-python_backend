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
// Unit Tests for scoped_defer.h/.cc
// Tests the ScopedDefer RAII class for deferred execution
//

#include <gtest/gtest.h>

#include "scoped_defer.h"

namespace triton { namespace backend { namespace python {
namespace test {

//
// Test ScopedDefer basic functionality
//
class ScopedDeferTest : public ::testing::Test {};

// Test that destructor executes the deferred task
TEST_F(ScopedDeferTest, DestructorExecutesTask)
{
  bool executed = false;

  {
    ScopedDefer defer([&executed]() { executed = true; });
    EXPECT_FALSE(executed);  // Should not execute yet
  }

  EXPECT_TRUE(executed);  // Should execute when scope ends
}

// Test that Complete() executes the task early
TEST_F(ScopedDeferTest, CompleteExecutesTaskEarly)
{
  bool executed = false;

  {
    ScopedDefer defer([&executed]() { executed = true; });
    EXPECT_FALSE(executed);

    defer.Complete();
    EXPECT_TRUE(executed);  // Should execute immediately
  }

  // Task should only execute once (already completed)
  EXPECT_TRUE(executed);
}

// Test that Complete() prevents double execution
TEST_F(ScopedDeferTest, CompletePreventDoubleExecution)
{
  int execution_count = 0;

  {
    ScopedDefer defer([&execution_count]() { execution_count++; });

    defer.Complete();
    EXPECT_EQ(execution_count, 1);

    // Second Complete() should not execute again
    defer.Complete();
    EXPECT_EQ(execution_count, 1);
  }

  // Destructor should not execute again either
  EXPECT_EQ(execution_count, 1);
}

// Test with no early completion - destructor handles it
TEST_F(ScopedDeferTest, DestructorOnlyExecution)
{
  int execution_count = 0;

  {
    ScopedDefer defer([&execution_count]() { execution_count++; });
    EXPECT_EQ(execution_count, 0);
    // Let it go out of scope naturally
  }

  EXPECT_EQ(execution_count, 1);
}

// Test with multiple defers in sequence
TEST_F(ScopedDeferTest, MultipleDefersInSequence)
{
  std::vector<int> execution_order;

  {
    ScopedDefer defer1([&execution_order]() { execution_order.push_back(1); });
    {
      ScopedDefer defer2(
          [&execution_order]() { execution_order.push_back(2); });
      {
        ScopedDefer defer3(
            [&execution_order]() { execution_order.push_back(3); });
        EXPECT_TRUE(execution_order.empty());
      }
      // defer3 should have executed
      ASSERT_EQ(execution_order.size(), 1u);
      EXPECT_EQ(execution_order[0], 3);
    }
    // defer2 should have executed
    ASSERT_EQ(execution_order.size(), 2u);
    EXPECT_EQ(execution_order[1], 2);
  }
  // defer1 should have executed
  ASSERT_EQ(execution_order.size(), 3u);
  EXPECT_EQ(execution_order[2], 1);
}

// Test with lambda capturing by reference
TEST_F(ScopedDeferTest, LambdaCaptureByReference)
{
  int value = 10;

  {
    ScopedDefer defer([&value]() { value = 42; });
    EXPECT_EQ(value, 10);
  }

  EXPECT_EQ(value, 42);
}

// Test with lambda capturing by value
TEST_F(ScopedDeferTest, LambdaCaptureByValue)
{
  int* ptr = new int(10);

  {
    ScopedDefer defer([ptr]() { delete ptr; });
    EXPECT_EQ(*ptr, 10);
  }

  // ptr is deleted, we can't access it anymore
  // This test verifies the delete happens (no memory leak)
}

// Test early completion with state modification
TEST_F(ScopedDeferTest, EarlyCompletionWithStateModification)
{
  std::string result;

  {
    ScopedDefer defer([&result]() { result += "deferred"; });
    result = "initial_";

    defer.Complete();
    EXPECT_EQ(result, "initial_deferred");

    result += "_after";
  }

  EXPECT_EQ(result, "initial_deferred_after");
}

// Test exception safety - task should still execute even if exception occurs
TEST_F(ScopedDeferTest, ExceptionSafety)
{
  bool executed = false;

  try {
    ScopedDefer defer([&executed]() { executed = true; });
    throw std::runtime_error("test exception");
  }
  catch (const std::runtime_error&) {
    // Exception caught
  }

  EXPECT_TRUE(executed);  // Deferred task should still have executed
}

// Test with empty/no-op lambda
TEST_F(ScopedDeferTest, NoOpLambda)
{
  {
    ScopedDefer defer([]() {});
    // Should not crash
  }
  // Should not crash on destruction
}

// Test Complete() called multiple times is safe
TEST_F(ScopedDeferTest, MultipleCompleteCallsSafe)
{
  int count = 0;

  {
    ScopedDefer defer([&count]() { count++; });

    defer.Complete();
    defer.Complete();
    defer.Complete();

    EXPECT_EQ(count, 1);
  }

  EXPECT_EQ(count, 1);
}

}  // namespace test
}}}  // namespace triton::backend::python

