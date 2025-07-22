#include <cuda_runtime.h>

#include "render.h"
#include "world.h"

__device__ Vec3 color(const Ray& ray, const World& world) {
    Hit hit;
    // Color anything that is in the world
    if (world.hit(ray, 0, Math::infinity, hit)) {
        return 0.5f * (hit.normal + Vec3{1, 1, 1});
    }

    // Lerped gradient for the background
    const auto unit_direction { Vec3::unit_vector(ray.direction()) };
    const auto a { 0.5f * (unit_direction.y() + 1.0f) };

    return (1.0 - a) * Vec3{1.0f, 1.0f, 1.0f} + a * Vec3{0.5f, 0.7f, 1.0f};
}

__global__ void render(const RenderContext* ctx, FrameBuffer* fb) {
    const auto c { blockIdx.x * blockDim.x + threadIdx.x };
    const auto r { blockIdx.y * blockDim.y + threadIdx.y };
    if( (r >= fb->width()) || (c >= fb->height()) ) return;

    const Vec3 pixel_center { ctx->first_pixel() + (c * ctx->pixel_delta_u()) + (r * ctx->pixel_delta_v()) };
    const Ray ray { ctx->camera_center(), pixel_center - ctx->camera_center() };

    fb->at(r, c) = color(ray, ctx->world());
}
