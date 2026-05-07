#include <cuda_runtime.h>

#include "render.h"

__global__ void init_rng(curandState* states, std::size_t width, std::size_t height) {
    const auto c { blockIdx.x * blockDim.x + threadIdx.x };
    const auto r { blockIdx.y * blockDim.y + threadIdx.y };
    if( (c >= width) || (r >= height) ) return;

    // Generates a seed for each index of states, i.e. each pixel/thread
    curand_init(r * width + c, 0, 0, &states[r * width + c]);
}



__device__ Vec3 color(const Ray& ray, const World* world) {
    if (const auto hit { world->hit(ray, {0, Math::infinity}) })
        return 0.5 * (hit->normal + Vec3{1.0, 1.0, 1.0});

    const auto unit_direction { Vec3::unit_vector(ray.direction()) };
    const auto a { 0.5f * (unit_direction.y() + 1.0f) };

    return (1.0 - a) * Vec3{1.0f, 1.0f, 1.0f} + a * Vec3{0.5f, 0.7f, 1.0f};
}

__global__ void render(RenderContext* ctx) {
    const auto c { blockIdx.x * blockDim.x + threadIdx.x };
    const auto r { blockIdx.y * blockDim.y + threadIdx.y };
    if( (c >= ctx->framebuffer->width()) || (r >= ctx->framebuffer->height()) ) return;

    Vec3 final_color {};
    auto& rng { ctx->rng[r * ctx->framebuffer->width() + c] };

    // TODO: probably a more efficient way to do this in parallel, instead of this serial loop; needs investigation
    for (uint32_t sample { 0 }; sample < ctx->samples_per_pixel; ++sample) {
        // Using per-thread seed, generate a random vector within the [-0.5, -0.5] to [0.5, 0.5] unit square
        const Vec3 offset {
            curand_uniform(&rng) - 0.5f,
            curand_uniform(&rng) - 0.5f,
            0
        };

        // Shoot ray at pixel +/- the small random offset
        const Vec3 pixel_sample {
            ctx->first_pixel
            + ((c + offset.x()) * ctx->pixel_delta_u)
            + ((r + offset.y()) * ctx->pixel_delta_v)
        };
        const Ray ray { ctx->camera_center, pixel_sample - ctx->camera_center };

        final_color += ctx->pixel_samples_scale * color(ray, ctx->world);
    }

    ctx->framebuffer->at(r, c) = final_color;
}
