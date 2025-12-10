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
// HIP IPC Child Process
// Separate executable that opens IPC handles passed from parent process.
// Used for testing IPC functionality which requires separate processes.
//
// Exit codes:
//   0 = Success
//   1 = Usage error
//   2 = Failed to open handle file
//   3 = Failed to read handle data
//   4 = hipSetDevice failed
//   5 = hipIpcOpenMemHandle / OpenHipHandle failed
//   6 = Data verification / memcpy failed
//   7 = hipIpcCloseMemHandle / CloseHipHandle failed
//   8 = HIPHandler not available
//

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>

#include "pb_exception.h"
#include "pb_utils.h"

using triton::backend::python::HIPHandler;
using triton::backend::python::PythonBackendException;

// Command modes - must match hip_ipc_test.cc and hip_handler_test.cc
enum class Mode {
  VERIFY,              // Open handle, verify data, close (raw HIP API)
  WRITE_BACK,          // Open handle, modify data, close (raw HIP API)
  HANDLER_VERIFY,      // Same as VERIFY but using HIPHandler
  HANDLER_WRITE_BACK,  // Same as WRITE_BACK but using HIPHandler
};

struct IPCChildParams {
  hipIpcMemHandle_t ipc_handle;
  int device_id;
  size_t num_elements;
  float expected_data[16];  // For verification
  Mode mode;
};

// Use raw HIP API for IPC operations
int RunWithRawHipApi(const IPCChildParams& params, void* mapped_ptr)
{
  hipError_t err;

  if (params.mode == Mode::VERIFY) {
    // Copy data from GPU to host and verify
    std::vector<float> host_data(params.num_elements);
    err = hipMemcpy(
        host_data.data(), mapped_ptr,
        params.num_elements * sizeof(float),
        hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << "\n";
      (void)hipIpcCloseMemHandle(mapped_ptr);
      return 6;
    }

    // Verify data
    for (size_t i = 0; i < params.num_elements; ++i) {
      if (host_data[i] != params.expected_data[i]) {
        std::cerr << "Data mismatch at index " << i << ": got "
                  << host_data[i] << ", expected " << params.expected_data[i]
                  << "\n";
        (void)hipIpcCloseMemHandle(mapped_ptr);
        return 6;
      }
    }
    std::cout << "Data verification passed\n";

  } else if (params.mode == Mode::WRITE_BACK) {
    // Read, modify (multiply by 2), write back
    std::vector<float> host_data(params.num_elements);
    err = hipMemcpy(
        host_data.data(), mapped_ptr,
        params.num_elements * sizeof(float),
        hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      std::cerr << "hipMemcpy (read) failed: " << hipGetErrorString(err) << "\n";
      (void)hipIpcCloseMemHandle(mapped_ptr);
      return 6;
    }

    // Modify data
    for (size_t i = 0; i < params.num_elements; ++i) {
      host_data[i] *= 2.0f;
    }

    // Write back
    err = hipMemcpy(
        mapped_ptr, host_data.data(),
        params.num_elements * sizeof(float),
        hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      std::cerr << "hipMemcpy (write) failed: " << hipGetErrorString(err) << "\n";
      (void)hipIpcCloseMemHandle(mapped_ptr);
      return 6;
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      std::cerr << "hipDeviceSynchronize failed: " << hipGetErrorString(err) << "\n";
      (void)hipIpcCloseMemHandle(mapped_ptr);
      return 6;
    }
    std::cout << "Write-back completed\n";
  }

  // Close IPC handle
  err = hipIpcCloseMemHandle(mapped_ptr);
  if (err != hipSuccess) {
    std::cerr << "hipIpcCloseMemHandle failed: " << hipGetErrorString(err) << "\n";
    return 7;
  }

  return 0;
}

// Use HIPHandler class for IPC operations
int RunWithHIPHandler(IPCChildParams& params)
{
  HIPHandler& handler = HIPHandler::getInstance();

  if (!handler.IsAvailable()) {
    std::cerr << "HIPHandler not available\n";
    return 8;
  }

  // Use HIPHandler to set device
  try {
    handler.MaybeSetDevice(params.device_id);
  } catch (const PythonBackendException& e) {
    std::cerr << "MaybeSetDevice failed: " << e.what() << "\n";
    return 4;
  }

  // Open IPC handle using HIPHandler
  void* mapped_ptr = nullptr;
  try {
    handler.OpenHipHandle(params.device_id, &params.ipc_handle, &mapped_ptr);
  } catch (const PythonBackendException& e) {
    std::cerr << "OpenHipHandle failed: " << e.what() << "\n";
    return 5;
  }

  if (mapped_ptr == nullptr) {
    std::cerr << "OpenHipHandle returned null pointer\n";
    return 5;
  }

  hipError_t err;

  if (params.mode == Mode::HANDLER_VERIFY) {
    // Copy data from GPU to host and verify
    std::vector<float> host_data(params.num_elements);
    err = hipMemcpy(
        host_data.data(), mapped_ptr,
        params.num_elements * sizeof(float),
        hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << "\n";
      try { handler.CloseHipHandle(params.device_id, mapped_ptr); }
      catch (...) {}
      return 6;
    }

    // Verify data
    for (size_t i = 0; i < params.num_elements; ++i) {
      if (host_data[i] != params.expected_data[i]) {
        std::cerr << "Data mismatch at index " << i << ": got "
                  << host_data[i] << ", expected " << params.expected_data[i]
                  << "\n";
        try { handler.CloseHipHandle(params.device_id, mapped_ptr); }
        catch (...) {}
        return 6;
      }
    }
    std::cout << "Data verification passed (HIPHandler)\n";

  } else if (params.mode == Mode::HANDLER_WRITE_BACK) {
    // Read, modify (multiply by 2), write back
    std::vector<float> host_data(params.num_elements);
    err = hipMemcpy(
        host_data.data(), mapped_ptr,
        params.num_elements * sizeof(float),
        hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      std::cerr << "hipMemcpy (read) failed: " << hipGetErrorString(err) << "\n";
      try { handler.CloseHipHandle(params.device_id, mapped_ptr); }
      catch (...) {}
      return 6;
    }

    for (size_t i = 0; i < params.num_elements; ++i) {
      host_data[i] *= 2.0f;
    }

    err = hipMemcpy(
        mapped_ptr, host_data.data(),
        params.num_elements * sizeof(float),
        hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      std::cerr << "hipMemcpy (write) failed: " << hipGetErrorString(err) << "\n";
      try { handler.CloseHipHandle(params.device_id, mapped_ptr); }
      catch (...) {}
      return 6;
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      std::cerr << "hipDeviceSynchronize failed: " << hipGetErrorString(err) << "\n";
      try { handler.CloseHipHandle(params.device_id, mapped_ptr); }
      catch (...) {}
      return 6;
    }
    std::cout << "Write-back completed (HIPHandler)\n";
  }

  // Close IPC handle using HIPHandler
  try {
    handler.CloseHipHandle(params.device_id, mapped_ptr);
  } catch (const PythonBackendException& e) {
    std::cerr << "CloseHipHandle failed: " << e.what() << "\n";
    return 7;
  }

  return 0;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "Usage: hip_ipc_child <params_file>\n";
    return 1;
  }

  const char* params_path = argv[1];

  // Read parameters from file
  std::ifstream in(params_path, std::ios::binary);
  if (!in) {
    std::cerr << "Failed to open params file: " << params_path << "\n";
    return 2;
  }

  IPCChildParams params;
  in.read(reinterpret_cast<char*>(&params), sizeof(params));

  if (!in) {
    std::cerr << "Failed to read params from file\n";
    return 3;
  }
  in.close();

  // Check if using HIPHandler modes
  if (params.mode == Mode::HANDLER_VERIFY ||
      params.mode == Mode::HANDLER_WRITE_BACK) {
    return RunWithHIPHandler(params);
  }

  // Raw HIP API path
  hipError_t err = hipSetDevice(params.device_id);
  if (err != hipSuccess) {
    std::cerr << "hipSetDevice(" << params.device_id << ") failed: "
              << hipGetErrorString(err) << "\n";
    return 4;
  }

  // Open the IPC handle
  void* mapped_ptr = nullptr;
  err = hipIpcOpenMemHandle(
      &mapped_ptr, params.ipc_handle, hipIpcMemLazyEnablePeerAccess);
  if (err != hipSuccess) {
    std::cerr << "hipIpcOpenMemHandle failed: " << hipGetErrorString(err)
              << " (error code: " << static_cast<int>(err) << ")\n";
    return 5;
  }

  return RunWithRawHipApi(params, mapped_ptr);
}
