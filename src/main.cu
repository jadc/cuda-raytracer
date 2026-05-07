#include <fstream>
#include <cuda_runtime.h>

#include "cuda.h"
#include "framebuffer.h"
#include "render.h"

int main() {
    FrameBuffer fb { 1280, 16.0 / 10.0 };

    World world(2);
    world.emplace<Sphere>(Vec3{ 0, 0, -1 }, 0.5f);
    world.emplace<Sphere>(Vec3{ 0, -100.5, -1 }, 100.0f);

    // Define number of blocks and threads
    constexpr std::size_t block_width  { 8 };  // in threads
    constexpr std::size_t block_height { 8 };  // in threads

    dim3 blocks {
        static_cast<uint32_t>(fb.width() / block_width + 1),
        static_cast<uint32_t>(fb.height() / block_height + 1),
    };
    dim3 threads { block_width, block_height };

    // Generate an RNG seed for each thread
    UnifiedMemory<curandState> rng(fb.width() * fb.height());
    init_rng<<<blocks, threads>>>(rng.get(), fb.width(), fb.height());
    cuda_unwrap(cudaGetLastError());
    cuda_unwrap(cudaDeviceSynchronize());

    auto ctx = RenderContext()
        .set_camera_center({ 0, 0, 0 })
        .set_focal_length(1.0f)
        .set_framebuffer(fb)
        .set_rng(rng.get())
        .set_samples_per_pixel(100)
        .set_viewport_height(2.0f)
        .set_world(world)
        .build();

    // Render from GPU into frame buffer
    render<<<blocks, threads>>>(&ctx);
    cuda_unwrap(cudaGetLastError());
    cuda_unwrap(cudaDeviceSynchronize());

    // Write frame buffer to ppm
    std::ofstream file { "output.ppm" };
    file << fb;
    file.close();
    return 0;
}
