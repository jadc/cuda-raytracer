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
