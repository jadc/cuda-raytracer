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

template <typename T, std::size_t N = 1>
class GlobalMemory {
    T* ptr;
public:
    template <typename... Args>
    __host__ GlobalMemory(Args&&... args) {
        // Allocate device global memory for T
        cuda_unwrap(cudaMalloc(static_cast<void**>(&ptr), N * sizeof(T)));

        // Create temporary based on constructor parameters
        T temp ( std::forward<Args>(args)... );

        // Initialize T in device global memory with temporary
        cuda_unwrap(cudaMemcpy(ptr, &temp, sizeof(T), cudaMemcpyHostToDevice));
    }

    __host__ ~GlobalMemory() {
        // TODO: Find a way to call the destructor on device automatically
        cuda_unwrap(cudaFree(ptr));
    }

    __device__ T& operator*() { return *ptr; }
    __host__ __device__ T* const get() const { return ptr; }
    __device__ T* get() { return ptr; }
    __device__ T* operator->() { return ptr; }
    __device__ const T& operator*() const { return *ptr; }
};

template <typename T>
class UnifiedMemory {
    T* ptr;
    std::size_t count = 1;
public:
    template <typename... Args>
    __host__ UnifiedMemory(std::size_t count, Args&&... args) : count{count} {
        // Allocate device global memory for T
        cuda_unwrap(cudaMallocManaged(&ptr, count * sizeof(T)));

        // Initialize T with constructor arguments
        new(ptr) T( std::forward<Args>(args)... );
    }

    __host__ ~UnifiedMemory() {
        // Call destructor through pointer on n element(s)
        for (std::size_t i = 0; i < count; ++i)
            ptr[i].~T();

        // Free memory from device
        cuda_unwrap(cudaFree(ptr));
    }

    __host__ __device__ T& operator*() { return *ptr; }
    __host__ __device__ T* const get() const { return ptr; }
    __host__ __device__ T* get() { return ptr; }
    __host__ __device__ T* operator->() { return ptr; }
    __host__ __device__ const T& operator*() const { return *ptr; }
};
