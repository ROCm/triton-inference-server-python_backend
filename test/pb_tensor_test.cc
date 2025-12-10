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
#include <vector>

#include "pb_tensor.h"
#include "pb_memory.h"
#include "pb_exception.h"
#include "shm_manager.h"

#ifdef TRITON_ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace triton { namespace backend { namespace python {
namespace test {

//
// PbTensor CPU Tests
//
class PbTensorCPUTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    shm_region_name_ = "/pb_tensor_test_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_,
        4 * 1024 * 1024,  // 4MB
        1024 * 1024,      // 1MB growth
        true);
  }

  void TearDown() override
  {
    shm_pool_.reset();
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
};

// Test creating a CPU tensor from raw pointer
TEST_F(PbTensorCPUTest, CreateFromRawPointer)
{
  std::string name = "test_tensor";
  std::vector<int64_t> dims = {2, 3, 4};
  std::vector<float> data(24, 1.0f);  // 2*3*4 = 24 elements

  auto tensor = std::make_unique<PbTensor>(
      name,
      dims,
      TRITONSERVER_TYPE_FP32,
      TRITONSERVER_MEMORY_CPU,
      0,  // memory_type_id
      data.data(),
      data.size() * sizeof(float));

  EXPECT_EQ(tensor->Name(), name);
  EXPECT_EQ(tensor->TritonDtype(), TRITONSERVER_TYPE_FP32);
  EXPECT_EQ(tensor->MemoryType(), TRITONSERVER_MEMORY_CPU);
  EXPECT_EQ(tensor->MemoryTypeId(), 0);
  EXPECT_EQ(tensor->ByteSize(), 24 * sizeof(float));
  EXPECT_TRUE(tensor->IsCPU());

  const auto& tensor_dims = tensor->Dims();
  ASSERT_EQ(tensor_dims.size(), 3);
  EXPECT_EQ(tensor_dims[0], 2);
  EXPECT_EQ(tensor_dims[1], 3);
  EXPECT_EQ(tensor_dims[2], 4);
}

// Test tensor name access
TEST_F(PbTensorCPUTest, NameAccess)
{
  std::vector<int64_t> dims = {10};
  std::vector<int32_t> data(10, 42);

  auto tensor = std::make_unique<PbTensor>(
      "test_tensor_name",
      dims,
      TRITONSERVER_TYPE_INT32,
      TRITONSERVER_MEMORY_CPU,
      0,
      data.data(),
      data.size() * sizeof(int32_t));

  EXPECT_EQ(tensor->Name(), "test_tensor_name");
}

// Test various data types
TEST_F(PbTensorCPUTest, DifferentDataTypes)
{
  std::vector<int64_t> dims = {4};

  // INT8
  {
    std::vector<int8_t> data = {1, 2, 3, 4};
    auto tensor = std::make_unique<PbTensor>(
        "int8_tensor", dims, TRITONSERVER_TYPE_INT8,
        TRITONSERVER_MEMORY_CPU, 0, data.data(), data.size());
    EXPECT_EQ(tensor->TritonDtype(), TRITONSERVER_TYPE_INT8);
    EXPECT_EQ(tensor->ByteSize(), 4);
  }

  // INT16
  {
    std::vector<int16_t> data = {1, 2, 3, 4};
    auto tensor = std::make_unique<PbTensor>(
        "int16_tensor", dims, TRITONSERVER_TYPE_INT16,
        TRITONSERVER_MEMORY_CPU, 0, data.data(), data.size() * sizeof(int16_t));
    EXPECT_EQ(tensor->TritonDtype(), TRITONSERVER_TYPE_INT16);
    EXPECT_EQ(tensor->ByteSize(), 8);
  }

  // INT64
  {
    std::vector<int64_t> data = {1, 2, 3, 4};
    auto tensor = std::make_unique<PbTensor>(
        "int64_tensor", dims, TRITONSERVER_TYPE_INT64,
        TRITONSERVER_MEMORY_CPU, 0, data.data(), data.size() * sizeof(int64_t));
    EXPECT_EQ(tensor->TritonDtype(), TRITONSERVER_TYPE_INT64);
    EXPECT_EQ(tensor->ByteSize(), 32);
  }

  // FP64 (double)
  {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    auto tensor = std::make_unique<PbTensor>(
        "fp64_tensor", dims, TRITONSERVER_TYPE_FP64,
        TRITONSERVER_MEMORY_CPU, 0, data.data(), data.size() * sizeof(double));
    EXPECT_EQ(tensor->TritonDtype(), TRITONSERVER_TYPE_FP64);
    EXPECT_EQ(tensor->ByteSize(), 32);
  }
}

// Test tensor with different dimensions
TEST_F(PbTensorCPUTest, DifferentDimensions)
{
  std::vector<float> data(120, 0.0f);

  // 1D tensor
  {
    std::vector<int64_t> dims = {120};
    auto tensor = std::make_unique<PbTensor>(
        "1d_tensor", dims, TRITONSERVER_TYPE_FP32,
        TRITONSERVER_MEMORY_CPU, 0, data.data(), data.size() * sizeof(float));
    EXPECT_EQ(tensor->Dims().size(), 1);
    EXPECT_EQ(tensor->Dims()[0], 120);
  }

  // 2D tensor
  {
    std::vector<int64_t> dims = {10, 12};
    auto tensor = std::make_unique<PbTensor>(
        "2d_tensor", dims, TRITONSERVER_TYPE_FP32,
        TRITONSERVER_MEMORY_CPU, 0, data.data(), data.size() * sizeof(float));
    EXPECT_EQ(tensor->Dims().size(), 2);
    EXPECT_EQ(tensor->Dims()[0], 10);
    EXPECT_EQ(tensor->Dims()[1], 12);
  }

  // 4D tensor (like image batches)
  {
    std::vector<int64_t> dims = {1, 3, 4, 10};  // batch, channels, height, width
    auto tensor = std::make_unique<PbTensor>(
        "4d_tensor", dims, TRITONSERVER_TYPE_FP32,
        TRITONSERVER_MEMORY_CPU, 0, data.data(), data.size() * sizeof(float));
    EXPECT_EQ(tensor->Dims().size(), 4);
  }
}

// Test SaveToSharedMemory and LoadFromSharedMemory for CPU tensor
TEST_F(PbTensorCPUTest, SaveAndLoadFromSharedMemory)
{
  std::string name = "shm_tensor";
  std::vector<int64_t> dims = {2, 4};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  // Create and save tensor
  auto original = std::make_unique<PbTensor>(
      name, dims, TRITONSERVER_TYPE_FP32,
      TRITONSERVER_MEMORY_CPU, 0,
      data.data(), data.size() * sizeof(float));

  original->SaveToSharedMemory(shm_pool_, false /* copy_gpu */);
  auto handle = original->ShmHandle();

  // Load tensor from shared memory
  auto loaded = PbTensor::LoadFromSharedMemory(
      shm_pool_, handle, false /* open_cuda_handle */);

  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->Name(), name);
  EXPECT_EQ(loaded->TritonDtype(), TRITONSERVER_TYPE_FP32);
  EXPECT_EQ(loaded->MemoryType(), TRITONSERVER_MEMORY_CPU);
  EXPECT_EQ(loaded->ByteSize(), data.size() * sizeof(float));

  const auto& loaded_dims = loaded->Dims();
  ASSERT_EQ(loaded_dims.size(), dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(loaded_dims[i], dims[i]);
  }

  // Verify data
  float* loaded_data = static_cast<float*>(loaded->DataPtr());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(loaded_data[i], data[i]);
  }
}

// Test empty tensor
TEST_F(PbTensorCPUTest, EmptyTensor)
{
  std::vector<int64_t> dims = {0};

  auto tensor = std::make_unique<PbTensor>(
      "empty_tensor", dims, TRITONSERVER_TYPE_FP32,
      TRITONSERVER_MEMORY_CPU, 0, nullptr, 0);

  EXPECT_EQ(tensor->ByteSize(), 0);
  EXPECT_EQ(tensor->Dims()[0], 0);
}

// Test DataPtr access
TEST_F(PbTensorCPUTest, DataPtrAccess)
{
  std::vector<int64_t> dims = {4};
  std::vector<int32_t> data = {10, 20, 30, 40};

  auto tensor = std::make_unique<PbTensor>(
      "data_ptr_tensor", dims, TRITONSERVER_TYPE_INT32,
      TRITONSERVER_MEMORY_CPU, 0,
      data.data(), data.size() * sizeof(int32_t));

  int32_t* ptr = static_cast<int32_t*>(tensor->DataPtr());
  EXPECT_EQ(ptr[0], 10);
  EXPECT_EQ(ptr[1], 20);
  EXPECT_EQ(ptr[2], 30);
  EXPECT_EQ(ptr[3], 40);
}

// Test MemoryType access
TEST_F(PbTensorCPUTest, MemoryTypeAccess)
{
  std::vector<int64_t> dims = {4};
  std::vector<float> data(4, 1.0f);

  auto tensor = std::make_unique<PbTensor>(
      "memory_type_tensor", dims, TRITONSERVER_TYPE_FP32,
      TRITONSERVER_MEMORY_CPU, 0,
      data.data(), data.size() * sizeof(float));

  EXPECT_EQ(tensor->MemoryType(), TRITONSERVER_MEMORY_CPU);
  EXPECT_EQ(tensor->MemoryTypeId(), 0);
}

#ifdef TRITON_ENABLE_ROCM

//
// PbTensor GPU Tests (ROCm-specific)
//
class PbTensorGPUTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }

    shm_region_name_ = "/pb_tensor_gpu_test_" + std::to_string(getpid());
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_,
        4 * 1024 * 1024,
        1024 * 1024,
        true);
  }

  void TearDown() override
  {
    for (void* ptr : gpu_allocations_) {
      (void)hipFree(ptr);
    }
    gpu_allocations_.clear();
    shm_pool_.reset();
  }

  void* AllocateGPUMemory(size_t size)
  {
    void* ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, size);
    if (err == hipSuccess && ptr != nullptr) {
      gpu_allocations_.push_back(ptr);
    }
    return ptr;
  }

  std::string shm_region_name_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
  std::vector<void*> gpu_allocations_;
};

// Test creating a GPU tensor
TEST_F(PbTensorGPUTest, CreateGPUTensor)
{
  std::string name = "gpu_tensor";
  std::vector<int64_t> dims = {4, 4};
  size_t byte_size = 16 * sizeof(float);

  void* gpu_ptr = AllocateGPUMemory(byte_size);
  ASSERT_NE(gpu_ptr, nullptr);

  auto tensor = std::make_unique<PbTensor>(
      name, dims, TRITONSERVER_TYPE_FP32,
      TRITONSERVER_MEMORY_GPU, 0,
      gpu_ptr, byte_size);

  EXPECT_EQ(tensor->Name(), name);
  EXPECT_EQ(tensor->TritonDtype(), TRITONSERVER_TYPE_FP32);
  EXPECT_EQ(tensor->MemoryType(), TRITONSERVER_MEMORY_GPU);
  EXPECT_EQ(tensor->MemoryTypeId(), 0);
  EXPECT_EQ(tensor->ByteSize(), byte_size);
  EXPECT_FALSE(tensor->IsCPU());
}

// Test GPU tensor with data initialization
TEST_F(PbTensorGPUTest, GPUTensorWithData)
{
  std::vector<int64_t> dims = {8};
  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  size_t byte_size = host_data.size() * sizeof(float);

  void* gpu_ptr = AllocateGPUMemory(byte_size);
  ASSERT_NE(gpu_ptr, nullptr);

  // Copy data to GPU
  hipError_t err = hipMemcpy(
      gpu_ptr, host_data.data(), byte_size, hipMemcpyHostToDevice);
  ASSERT_EQ(err, hipSuccess);

  auto tensor = std::make_unique<PbTensor>(
      "gpu_data_tensor", dims, TRITONSERVER_TYPE_FP32,
      TRITONSERVER_MEMORY_GPU, 0,
      gpu_ptr, byte_size);

  EXPECT_FALSE(tensor->IsCPU());
  EXPECT_EQ(tensor->DataPtr(), gpu_ptr);

  // Verify by copying back
  std::vector<float> verify_data(8);
  err = hipMemcpy(
      verify_data.data(), tensor->DataPtr(), byte_size, hipMemcpyDeviceToHost);
  ASSERT_EQ(err, hipSuccess);

  for (size_t i = 0; i < host_data.size(); ++i) {
    EXPECT_FLOAT_EQ(verify_data[i], host_data[i]);
  }
}

// Test SaveToSharedMemory for GPU tensor
TEST_F(PbTensorGPUTest, SaveGPUTensorToSharedMemory)
{
  std::vector<int64_t> dims = {4};
  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
  size_t byte_size = host_data.size() * sizeof(float);

  void* gpu_ptr = AllocateGPUMemory(byte_size);
  ASSERT_NE(gpu_ptr, nullptr);

  hipError_t err = hipMemcpy(
      gpu_ptr, host_data.data(), byte_size, hipMemcpyHostToDevice);
  ASSERT_EQ(err, hipSuccess);

  auto tensor = std::make_unique<PbTensor>(
      "gpu_shm_tensor", dims, TRITONSERVER_TYPE_FP32,
      TRITONSERVER_MEMORY_GPU, 0,
      gpu_ptr, byte_size);

  // Save to shared memory (with GPU IPC handle)
  tensor->SaveToSharedMemory(shm_pool_, true /* copy_gpu */);
  auto handle = tensor->ShmHandle();
  ASSERT_NE(handle, 0);

  // Validate tensor that was saved to shared memory
  // Note: open_cuda_handle must be false when loading in the same process
  // because HIP IPC handles are designed for inter-process communication.
  // Opening an IPC handle in the same process that created it results in
  // "invalid device context" error.
  auto loaded = PbTensor::LoadFromSharedMemory(
      shm_pool_, handle, false /* open_cuda_handle */);
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->Name(), "gpu_shm_tensor");
  EXPECT_EQ(loaded->TritonDtype(), TRITONSERVER_TYPE_FP32);
  EXPECT_EQ(loaded->MemoryType(), TRITONSERVER_MEMORY_GPU);
  EXPECT_EQ(loaded->ByteSize(), byte_size);
  EXPECT_FALSE(loaded->IsCPU());

  std::vector<float> verify_data(host_data.size());
  err = hipMemcpy(
      verify_data.data(), gpu_ptr, byte_size, hipMemcpyDeviceToHost);
  ASSERT_EQ(err, hipSuccess);

  for (size_t i = 0; i < host_data.size(); ++i) {
    EXPECT_FLOAT_EQ(verify_data[i], host_data[i]);
  }
}

#endif  // TRITON_ENABLE_ROCM

}  // namespace test
}}}  // namespace triton::backend::python

