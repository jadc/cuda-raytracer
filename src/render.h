#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "framebuffer.h"
#include "world.h"

struct RenderContext {
    FrameBuffer* framebuffer {};
    const World* world {};
    curandState* rng {};

    // Camera properties
    float focal_length {};
    float viewport_width {};
    float viewport_height {};

    // Multisampling for anti-aliasing
    uint32_t samples_per_pixel {};
    float pixel_samples_scale {};

    // Rays will be emitted from the camera center
    Vec3 camera_center;

    // Vectors that run along the edges of the viewport
    Vec3 viewport_u;
    Vec3 viewport_v;

    // Vectors representing distance between pixels
    Vec3 pixel_delta_u;
    Vec3 pixel_delta_v;

    // Vector pointing to upper left corner of viewport
    Vec3 viewport_upper_left;

    // Vector pointing to upper left pixel
    Vec3 first_pixel;

    __host__ RenderContext& set_camera_center(Vec3 c)         { camera_center = std::move(c); return *this; }
    __host__ RenderContext& set_focal_length(float f)         { focal_length = f;             return *this; }
    __host__ RenderContext& set_framebuffer(FrameBuffer& fb)  { framebuffer = &fb;            return *this; }
    __host__ RenderContext& set_rng(curandState* s)           { rng = s;                      return *this; }
    __host__ RenderContext& set_samples_per_pixel(uint32_t s) { samples_per_pixel = s;        return *this; }
    __host__ RenderContext& set_viewport_height(float h)      { viewport_height = h;          return *this; }
    __host__ RenderContext& set_world(const World& w)         { world = &w;                   return *this; }

    __host__ RenderContext build() {
        viewport_width = viewport_height * (static_cast<float>(framebuffer->width()) / framebuffer->height());
        viewport_u = { viewport_width, 0, 0 };
        viewport_v = { 0, -viewport_height, 0 };
        pixel_delta_u = viewport_u / framebuffer->width();
        pixel_delta_v = viewport_v / framebuffer->height();
        viewport_upper_left = camera_center - Vec3{0, 0, focal_length} - viewport_u/2 - viewport_v/2;
        first_pixel = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);
        pixel_samples_scale = 1.0f / samples_per_pixel;
        return *this;
    }
};

__global__ void init_rng(curandState* states, std::size_t width, std::size_t height);
__global__ void render(RenderContext* ctx);
