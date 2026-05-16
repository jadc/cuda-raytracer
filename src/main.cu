#include <fstream>
#include <cuda_runtime.h>

#include "cuda.h"
#include "framebuffer.h"
#include "material.h"
#include "render.h"

int main() {
    FrameBuffer fb { 1280, 16.0 / 10.0 };

    MaterialTable materials(6);
    auto* mat_ground { materials.emplace<Lambertian>(Vec3{0.4f, 0.2f, 0.5f}) };
    auto* mat_silver { materials.emplace<Metal>(Vec3{0.9f, 0.9f, 0.9f}) };
    auto* mat_gold   { materials.emplace<Metal>(Vec3{0.8f, 0.6f, 0.2f}) };
    auto* mat_red    { materials.emplace<Lambertian>(Vec3{0.9f, 0.1f, 0.1f}) };
    auto* mat_blue   { materials.emplace<Lambertian>(Vec3{0.1f, 0.2f, 0.9f}) };
    auto* mat_green  { materials.emplace<Lambertian>(Vec3{0.2f, 0.8f, 0.2f}) };

    World world(8);
    world.emplace<Sphere>(Vec3{ 0.0f, -100.5f, -1.0f}, 100.0f, mat_ground);
    world.emplace<Sphere>(Vec3{ 0.0f,  0.0f, -1.0f}, 0.5f, mat_silver);
    world.emplace<Sphere>(Vec3{-1.0f, -0.2f, -1.0f}, 0.3f, mat_gold);
    world.emplace<Sphere>(Vec3{ 1.0f, -0.2f, -1.0f}, 0.3f, mat_gold);
    world.emplace<Sphere>(Vec3{-0.4f, -0.35f, -0.6f}, 0.15f, mat_red);
    world.emplace<Sphere>(Vec3{ 0.4f, -0.35f, -0.6f}, 0.15f, mat_red);
    world.emplace<Sphere>(Vec3{-0.85f, -0.425f, -0.65f}, 0.075f, mat_blue);
    world.emplace<Sphere>(Vec3{ 0.85f, -0.425f, -0.65f}, 0.075f, mat_green);

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

    // Ensure enough stack size for recursive ray bounces
    cuda_unwrap(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

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
