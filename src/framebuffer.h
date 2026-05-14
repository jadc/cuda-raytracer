#pragma once

#include <cuda_runtime.h>
#include <ostream>
#include "math.h"
#include "cuda.h"

class FrameBuffer {
    std::size_t m_width;
    std::size_t m_height;
    UnifiedMemory<Vec3> m_pixels;
public:
    // Construct an empty frame buffer, allocating pixels on device
    __host__ FrameBuffer(std::size_t width, std::size_t height)
        : m_width{width}
        , m_height{height}
        , m_pixels{width * height}
        {}


    __host__ FrameBuffer(std::size_t width, double aspect_ratio)
        : FrameBuffer(width, std::max(static_cast<std::size_t>(width / aspect_ratio), 1lu))
        {}

    __host__ __device__ std::size_t width()  const { return m_width; }
    __host__ __device__ std::size_t height() const { return m_height; }

    __host__ __device__ Vec3 at(std::size_t row, std::size_t col) const {
        return m_pixels.get()[row * width() + col];
    }
    __host__ __device__ Vec3& at(std::size_t row, std::size_t col) {
        return m_pixels.get()[row * width() + col];
    }
};

constexpr float linear_to_gamma(float x) {
    return x > 0.0f ? std::sqrt(x) : 0.0f;
}

// Writes a frame buffer of RGB floats into the given stream as PPM
inline std::ostream& operator<<(std::ostream& os, const FrameBuffer& fb) {
    os << "P3\n" << fb.width() << ' ' << fb.height() << "\n255\n";

    for (std::size_t r { 0 }; r < fb.height(); ++r) {
        for (std::size_t c { 0 }; c < fb.width(); ++c) {
            const auto& pixel { fb.at(r, c) };
            auto r = pixel.x(), g = pixel.y(), b = pixel.z();

            // Convert from linear to gamma space
            r = linear_to_gamma(r), g = linear_to_gamma(g), b = linear_to_gamma(b);

            // Normalize vector components, then convert into RGB
            static constexpr Interval intensity { 0.000f, 0.999f };
            const auto r_byte { static_cast<unsigned>(256 * intensity.clamp(r)) };
            const auto g_byte { static_cast<unsigned>(256 * intensity.clamp(g)) };
            const auto b_byte { static_cast<unsigned>(256 * intensity.clamp(b)) };

            // Output pixel to stream
            os << r_byte << ' ' << g_byte << ' ' << b_byte << '\n';
        }
    }

    return os;
}
