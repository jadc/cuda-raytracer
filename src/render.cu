#include <cuda_runtime.h>

#include "render.h"

__device__ bool hit_sphere(const Ray& ray, const Vec3& sphere_center, float sphere_radius) {
    const Vec3 oc { sphere_center - ray.origin() };
    const auto a { Vec3::dot(ray.direction(), ray.direction()) };
    const auto b { -2.0f * Vec3::dot(ray.direction(), oc) };
    const auto c { Vec3::dot(oc, oc) - sphere_radius * sphere_radius };
    const auto discriminant { b * b - 4.0f * a * c };
    return discriminant >= 0;
}

__device__ Vec3 color(const Ray& ray) {
    // If ray intersects with sphere, draw green
    if (hit_sphere(ray, { 0, 0, -1 }, 0.5f))
        return { 0, 1, 0 };

    // Lerped gradient for the background
    const auto unit_direction { Vec3::unit_vector(ray.direction()) };
    const auto a { 0.5f * (unit_direction.y() + 1.0f) };

    return (1.0 - a) * Vec3{1.0f, 1.0f, 1.0f} + a * Vec3{0.5f, 0.7f, 1.0f};
}

__global__ void render(FrameBuffer* fb, const RenderContext* ctx) {
    const auto c { blockIdx.x * blockDim.x + threadIdx.x };
    const auto r { blockIdx.y * blockDim.y + threadIdx.y };
    if( (r >= fb->width()) || (c >= fb->height()) ) return;

    const Vec3 pixel_center { ctx->first_pixel() + (c * ctx->pixel_delta_u()) + (r * ctx->pixel_delta_v()) };
    const Ray ray { ctx->camera_center(), pixel_center - ctx->camera_center() };

    fb->at(r, c) = color(ray);
}
