#pragma once

#include <cuda_runtime.h>
#include "framebuffer.h"
#include "world.h"

struct RenderContext {
    const World* world;

    // Camera properties
    float focal_length;
    float viewport_width;
    float viewport_height;

    // Rays will be emitted from the camera center
    const Vec3 camera_center;

    // Vectors that run along the edges of the viewport
    const Vec3 viewport_u;
    const Vec3 viewport_v;

    // Vectors representing distance between pixels
    const Vec3 pixel_delta_u;
    const Vec3 pixel_delta_v;

    // Vector pointing to upper left corner of viewport
    const Vec3 viewport_upper_left;

    // Vector pointing to upper left pixel
    const Vec3 first_pixel;

    __host__ __device__ RenderContext(const FrameBuffer& fb, const World& world, float focal_length, float viewport_height, Vec3 camera_center)
        : focal_length { focal_length }
        , viewport_height { viewport_height }
        , viewport_width { viewport_height * (static_cast<float>(fb.width()) / fb.height()) }
        , camera_center { std::move(camera_center) }
        , viewport_u { viewport_width, 0, 0 }
        , viewport_v { 0, -viewport_height, 0 }
        , pixel_delta_u { viewport_u / fb.width() }
        , pixel_delta_v { viewport_v / fb.height() }
        , viewport_upper_left { camera_center - Vec3{0, 0, focal_length} - viewport_u/2 - viewport_v/2 }
        , first_pixel { viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v) }
        , world { &world }
    {};
};

__global__ void render(const RenderContext* ctx, FrameBuffer* fb);
