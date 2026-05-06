#include <fstream>
#include <cuda_runtime.h>

#include "cuda.h"
#include "framebuffer.h"
#include "render.h"

int main() {
    constexpr std::size_t width  { 512 };
    constexpr std::size_t height { 512 };

    FrameBuffer fb { width, height };

    World world(2);
    world.emplace<Sphere>(Vec3{ 0, 0, -1 }, 0.5f);
    world.emplace<Sphere>(Vec3{ 0, -100.5, -1 }, 100.0f);

    const RenderContext ctx {
        /*     framebuffer=*/fb,
        /*          world=*/world,
        /*    focal_length=*/1.0f,
        /* viewport_height=*/2.0f,
        /*   camera_center=*/{ 0, 0, 0 },
    };

    // Define number of blocks and threads
    constexpr std::size_t block_width  { 8 };  // in threads
    constexpr std::size_t block_height { 8 };  // in threads

    dim3 blocks {
        width / block_width + 1,
        height / block_height + 1,
    };
    dim3 threads { block_width, block_height };

    // Render from GPU into frame buffer
    render<<<blocks, threads>>>(&ctx, &fb);

    // Check for errors and synchronize
    cuda_unwrap(cudaGetLastError());
    cuda_unwrap(cudaDeviceSynchronize());

    // Write frame buffer to ppm
    std::ofstream file { "output.ppm" };
    file << fb;
    file.close();
    return 0;
}
