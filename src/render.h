#pragma once

#include <cuda_runtime.h>

#include "linalg.h"
#include "framebuffer.h"

class RenderContext {
    // Camera properties
    float m_focal_length;
    float m_viewport_width;
    float m_viewport_height;

    // Rays will be emitted from the camera center
    const Vec3 m_camera_center;

    // Vectors that run along the edges of the viewport
    const Vec3 m_viewport_u;
    const Vec3 m_viewport_v;

    // Vectors representing distance between pixels
    const Vec3 m_pixel_delta_u;
    const Vec3 m_pixel_delta_v;

    // Vector pointing to upper left corner of viewport
    const Vec3 m_viewport_upper_left;

    // Vector pointing to upper left pixel
    const Vec3 m_first_pixel;

public:
    __host__ __device__ RenderContext(const FrameBuffer& fb, float focal_length, float viewport_height, Vec3 camera_center)
        : m_focal_length { focal_length }
        , m_viewport_height { viewport_height }
        , m_viewport_width { viewport_height * (static_cast<float>(fb.width()) / fb.height()) }
        , m_camera_center { std::move(camera_center) }
        , m_viewport_u { m_viewport_width, 0, 0 }  // horizontal
        , m_viewport_v { 0, -viewport_height, 0 }  // vertical
        , m_pixel_delta_u { m_viewport_u / fb.width() }   // horizontal
        , m_pixel_delta_v { m_viewport_v / fb.height() }  // vertical
        , m_viewport_upper_left { camera_center - Vec3{0, 0, focal_length} - m_viewport_u/2 - m_viewport_v/2 }
        , m_first_pixel { m_viewport_upper_left + 0.5f * (m_pixel_delta_u + m_pixel_delta_v) }
    {};

    __device__ const Vec3& camera_center() const { return m_camera_center; };
    __device__ const Vec3& first_pixel()   const { return m_first_pixel; };
    __device__ const Vec3& pixel_delta_u() const { return m_pixel_delta_u; };
    __device__ const Vec3& pixel_delta_v() const { return m_pixel_delta_v; };
};

__global__ void render(FrameBuffer* fb, const RenderContext* ctx);
