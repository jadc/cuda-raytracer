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

template <typename T>
class GlobalMemory {
    T* ptr;
public:
    __host__ GlobalMemory(std::size_t count = 1) {
        cuda_unwrap(cudaMalloc(&ptr, count * sizeof(T)));
    }

    __host__ ~GlobalMemory() {
        if (ptr) cuda_unwrap(cudaFree(ptr));
    }

    // Disable copy constructors
    GlobalMemory(const GlobalMemory&) = delete;
    GlobalMemory& operator=(const GlobalMemory&) = delete;

    // Move constructors
    GlobalMemory(GlobalMemory&& other) noexcept : ptr{other.ptr} {
        other.ptr = nullptr;
    }
    GlobalMemory& operator=(GlobalMemory&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    __device__ T& operator[](std::size_t i) { return ptr[i]; }
    __device__ const T& operator[](std::size_t i) const { return ptr[i]; }
    __host__ __device__ T* get() { return ptr; }
    __host__ __device__ const T* get() const { return ptr; }
};

template <typename T>
class UnifiedMemory {
    T* ptr;
public:
    __host__ UnifiedMemory(std::size_t count = 1) {
        cuda_unwrap(cudaMallocManaged(&ptr, count * sizeof(T)));
    }

    __host__ ~UnifiedMemory() {
        if (ptr) cuda_unwrap(cudaFree(ptr));
    }

    // Disable copy constructors
    UnifiedMemory(const UnifiedMemory&) = delete;
    UnifiedMemory& operator=(const UnifiedMemory&) = delete;

    // Move constructors
    UnifiedMemory(UnifiedMemory&& other) noexcept : ptr{other.ptr} {
        other.ptr = nullptr;
    }
    UnifiedMemory& operator=(UnifiedMemory&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    __host__ __device__ T& operator[](std::size_t i) { return ptr[i]; }
    __host__ __device__ const T& operator[](std::size_t i) const { return ptr[i]; }
    __host__ __device__ T* get() { return ptr; }
    __host__ __device__ const T* get() const { return ptr; }
};
