#include <fstream>
#include <cuda_runtime.h>

#include "cuda.h"
#include "framebuffer.h"
#include "render.h"

int main() {
    FrameBuffer fb { /*width=*/512, /*height=*/512 };

    constexpr int block_width  { 8 };  // in threads
    constexpr int block_height { 8 };  // in threads

    // Define camera properties
    constexpr auto focal_length { 1.0f };
    constexpr auto viewport_height { 2.0f };
    const auto viewport_width { viewport_height * (static_cast<float>(fb.width()) / fb.height()) };

    // Rays will be emitted from the camera center
    const Vec3 camera_center { 0, 0, 0 };

    // Vectors that run along the edges of the viewport
    const Vec3 viewport_u { viewport_width, 0, 0 };    // horizontal
    const Vec3 viewport_v { 0, -viewport_height, 0 };  // vertical

    // Vectors representing distance between pixels
    const Vec3 pixel_delta_u { viewport_u / fb.width() };   // horizontal
    const Vec3 pixel_delta_v { viewport_u / fb.height() };  // vertical

    // Vector pointing to upper left pixel
    const Vec3 viewport_upper_left { camera_center - Vec3{0, 0, focal_length} - viewport_u/2 - viewport_v/2 };
    const Vec3 first_pixel { viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v) };

    // Define number of blocks and threads
    dim3 blocks {
        static_cast<unsigned int>(fb.width()) / block_width + 1,
        static_cast<unsigned int>(fb.height()) / block_height + 1,
    };
    dim3 threads { block_width, block_height };

    // Render from GPU into frame buffer
    render<<<blocks, threads>>>(&fb, &camera_center, &first_pixel, &pixel_delta_u, &pixel_delta_v);

    // Check for errors and synchronize
    cuda_unwrap(cudaGetLastError());
    cuda_unwrap(cudaDeviceSynchronize());

    // Write frame buffer to ppm
    std::ofstream file { "output.ppm" };
    file << fb;
    file.close();
    return 0;
}
