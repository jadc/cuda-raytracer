#include <fstream>
#include <cuda_runtime.h>

#include "cuda.h"
#include "vec3.h"
#include "framebuffer.h"

__global__ void hello_from_gpu() {
    printf("GPU thread %d\n", threadIdx.x);
}

template <std::size_t width, std::size_t height>
__global__ void render(FrameBuffer<width, height>* fb) {
    const auto c { blockIdx.x * blockDim.x + threadIdx.x };
    const auto r { blockIdx.y * blockDim.y + threadIdx.y };
    if( (r >= width) || (c >= height) ) return;

    fb->at(r, c) = {
        static_cast<float>(c) / height,
        static_cast<float>(r) / width,
        0.2f,
    };
}

int main() {
    constexpr int block_width  { 8 };  // in threads
    constexpr int block_height { 8 };  // in threads

    // Allocate unified memory (shared between host and device) for frame buffer
    FrameBuffer<256, 256> fb {};
    cuda_unwrap(cudaMallocManaged((void **)&fb, fb.width() * fb.height() * sizeof(Vec3)));

    // Define number of blocks and threads
    dim3 blocks {
        fb.width() / block_width + 1,
        fb.height() / block_height + 1,
    };
    dim3 threads { block_width, block_height };

    // Render from GPU into frame buffer
    render<<<blocks, threads>>>(&fb);

    // Check for errors and synchronize
    cuda_unwrap(cudaGetLastError());
    cuda_unwrap(cudaDeviceSynchronize());

    // Write frame buffer to ppm
    std::ofstream file { "output.ppm" };
    file << fb;
    file.close();
    return 0;
}
