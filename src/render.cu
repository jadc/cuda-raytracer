#include <cuda_runtime.h>

#include "render.h"

__global__ void render(FrameBuffer* fb, const Vec3* camera_center, const Vec3* first_pixel, const Vec3* pixel_delta_u, const Vec3* pixel_delta_v) {
    const auto c { blockIdx.x * blockDim.x + threadIdx.x };
    const auto r { blockIdx.y * blockDim.y + threadIdx.y };
    if( (r >= fb->width()) || (c >= fb->height()) ) return;

    const Vec3 pixel_center { *first_pixel + (c * *pixel_delta_u) + (r * *pixel_delta_v) };
    const Ray ray { *camera_center, pixel_center - *camera_center };

    // Lerped gradient for the background
    const auto unit_direction { Vec3::unit_vector(ray.direction()) };
    const auto a { 0.5f * (unit_direction.y() + 1.0f) };

    fb->at(r, c) = (1.0 - a) * Vec3{1.0f, 1.0f, 1.0f} + a * Vec3{0.3f, 0.5f, 1.0f};
}
