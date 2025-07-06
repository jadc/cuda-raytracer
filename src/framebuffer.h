#pragma once

#include <cuda_runtime.h>
#include <ostream>
#include <array>
#include "linalg.h"

template <std::size_t w, std::size_t h>
class FrameBuffer {
    Vec3 m_pixels[w * h];
public:
    // Construct an empty frame buffer
    __host__ __device__ FrameBuffer()
        : m_pixels{}
        {}

    // Construct a frame buffer with n vectors
    template<typename... Args>
    __host__ __device__ FrameBuffer(Args... args)
        : m_pixels{args...}
        {}

    __host__ __device__ constexpr std::size_t width()  const { return w; }
    __host__ __device__ constexpr std::size_t height() const { return h; }

    __host__ __device__ Vec3 at(std::size_t row, std::size_t col) const {
        return m_pixels[row * width() + col];
    }
    __host__ __device__ Vec3& at(std::size_t row, std::size_t col) {
        return m_pixels[row * width() + col];
    }
};

// Writes a frame buffer of RGB floats into the given stream as PPM
template <std::size_t width, std::size_t height>
inline std::ostream& operator<<(std::ostream& os, FrameBuffer<width, height>& fb) {
    os << "P3\n" << width << ' ' << height << "\n255\n";

    for (std::size_t c { 0 }; c < height; ++c) {
        for (std::size_t r { 0 }; r < width; ++r) {
            const auto& pixel { fb.at(r, c) };

            // Convert normalized vector components into RGB
            const auto ir { static_cast<int>(255.999 * pixel.x()) };
            const auto ig { static_cast<int>(255.999 * pixel.y()) };
            const auto ib { static_cast<int>(255.999 * pixel.z()) };

            // Output pixel to stream
            os << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    return os;
}

