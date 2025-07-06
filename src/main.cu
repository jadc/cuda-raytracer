#include <fstream>
#include <cuda_runtime.h>

#include "cuda.h"
#include "linalg.h"
#include "framebuffer.h"

// TODO: move all these parameters into their own object
template <std::size_t width, std::size_t height>
__global__ void render(FrameBuffer<width, height>* fb, const Vec3* camera_center, const Vec3* first_pixel, const Vec3* pixel_delta_u, const Vec3* pixel_delta_v) {
    const auto c { blockIdx.x * blockDim.x + threadIdx.x };
    const auto r { blockIdx.y * blockDim.y + threadIdx.y };
    if( (r >= fb->width()) || (c >= fb->height()) ) return;

    const Vec3 pixel_center { *first_pixel + (c * *pixel_delta_u) + (r * *pixel_delta_v) };
    const Ray ray { *camera_center, pixel_center - *camera_center };

    // Compute color of pixel that ray hit (lerped gradient)
    const auto unit_direction { Vec3::unit_vector(ray.direction()) };
    const auto a { 0.5f * (unit_direction.y() + 1.0f) };

    fb->at(r, c) = (1.0 - a) * Vec3{1.0f, 1.0f, 1.0f} + a * Vec3{0.3f, 0.5f, 1.0f};
}

int main() {
    constexpr int block_width  { 8 };  // in threads
    constexpr int block_height { 8 };  // in threads

    // Allocate unified memory (shared between host and device) for frame buffer
    FrameBuffer<512, 512> fb {};
    cuda_unwrap(cudaMallocManaged((void **)&fb, fb.width() * fb.height() * sizeof(Vec3)));

    // TODO: move all this math to a dedicated function
    // Define camera properties
    constexpr auto focal_length { 1.0f };
    constexpr auto viewport_height { 2.0f };
    constexpr auto viewport_width { viewport_height * (static_cast<float>(fb.width()) / fb.height()) };

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
        fb.width() / block_width + 1,
        fb.height() / block_height + 1,
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
