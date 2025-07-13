#include <fstream>
#include <cuda_runtime.h>

#include "cuda.h"
#include "framebuffer.h"
#include "render.h"

int main() {
    FrameBuffer fb { /*width=*/512, /*height=*/512 };

    // Define constants related to rendering
    const RenderContext ctx {
        /*     framebuffer=*/fb,
        /*    focal_length=*/1.0f,
        /* viewport_height=*/2.0f,
        /*   camera_center=*/{ 0, 0, 0 },
    };

    // Define number of blocks and threads
    constexpr int block_width  { 8 };  // in threads
    constexpr int block_height { 8 };  // in threads

    dim3 blocks {
        static_cast<unsigned int>(fb.width()) / block_width + 1,
        static_cast<unsigned int>(fb.height()) / block_height + 1,
    };
    dim3 threads { block_width, block_height };

    // Render from GPU into frame buffer
    render<<<blocks, threads>>>(&fb, &ctx);

    // Check for errors and synchronize
    cuda_unwrap(cudaGetLastError());
    cuda_unwrap(cudaDeviceSynchronize());

    // Write frame buffer to ppm
    std::ofstream file { "output.ppm" };
    file << fb;
    file.close();
    return 0;
}
