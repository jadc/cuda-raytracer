#pragma once

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for descriptive CUDA error logging and device cleanup
// A macro is used to encode information unique to the invocation:
// - the subroutine's source code (via #input)
// - the file it was called in (via __FILE__)
// - the line number in said file (via __LINE__)
// This is inspired by https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
#define cuda_unwrap(input) unwrap_cuda( (input), #input, __FILE__, __LINE__ )

inline void unwrap_cuda(cudaError_t error, const std::string func, const std::string file, const int line) {
    if (error == 0) return;
    std::cerr << file << "(" << line << "): CUDA error " << static_cast<unsigned int>(error)
    << " in '" << func << "'" << std::endl;

    cudaDeviceReset();
    std::exit(1);
}

// TODO: maybe make_global and make_unified?, both return DeviceMemory instances?

template <typename T>
class GlobalMemory {
    T* ptr;
public:
    template <typename... Args>
    __host__ GlobalMemory(Args&&... args) {
        // Allocate device global memory for T
        cuda_unwrap(cudaMalloc(static_cast<void**>(&ptr), sizeof(T)));
        cuda_unwrap(cudaGetLastError());
        cuda_unwrap(cudaDeviceSynchronize());

        // Create temporary based on constructor parameters
        // TODO: this is gonna call the destructor early
        T temp { std::forward<Args>(args)... };

        // Initialize T in device global memory with temporary
        cuda_unwrap(cudaMemcpy(ptr, &temp, sizeof(T), cudaMemcpyHostToDevice));
        cuda_unwrap(cudaGetLastError());
        cuda_unwrap(cudaDeviceSynchronize());
    }

    __host__ __device__ T* const get() const { return ptr; }
    __device__ T& operator*() { return *ptr; }
    __device__ const T& operator*() const { return *ptr; }
    __device__ T* operator->() { return ptr; }

    __host__ ~GlobalMemory() {
        cuda_unwrap(cudaDeviceSynchronize());
        cuda_unwrap(cudaFree(ptr));
        cuda_unwrap(cudaGetLastError());
    }
};


template <typename T>
class UnifiedMemory {
    T* ptr;
public:
    template <typename... Args>
    __host__ UnifiedMemory(Args&&... args) {
        // Allocate device global memory for T
        cuda_unwrap(cudaMallocManaged(&ptr, sizeof(T)));
        cuda_unwrap(cudaGetLastError());
        cuda_unwrap(cudaDeviceSynchronize());

        // Initialize T with constructor arguments
        new(ptr) T{ std::forward<Args>(args)... };
    }

    __host__ __device__ T* const get() const { return ptr; }
    __host__ __device__ T& operator*() { return *ptr; }
    __host__ __device__ const T& operator*() const { return *ptr; }
    __host__ __device__ T* operator->() { return ptr; }

    __host__ ~UnifiedMemory() {
        //delete ptr;
        cuda_unwrap(cudaDeviceSynchronize());
        cuda_unwrap(cudaFree(ptr));
        cuda_unwrap(cudaGetLastError());
    }
};
