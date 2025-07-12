#pragma once

#include <cuda_runtime.h>
#include <ostream>
#include <array>
#include "linalg.h"
#include "cuda.h"

class FrameBuffer {
    std::size_t m_width;
    std::size_t m_height;
    Vec3* m_pixels;
public:
    // Construct an empty frame buffer, allocating space on device
    __host__ FrameBuffer(std::size_t width, std::size_t height) : m_width{width}, m_height{height} {
        // Allocate unified memory (shared between host and device) for frame buffer
        // TODO: Managed memory is simpler, but not strictly necessary; replace with traditional device-only memory?
        cuda_unwrap(cudaMallocManaged(&m_pixels, width * height * sizeof(Vec3)));
    }

    // RAII
    __host__ ~FrameBuffer() {
        cuda_unwrap(cudaFree(m_pixels));
    }

    __host__ __device__ std::size_t width()  const { return m_width; }
    __host__ __device__ std::size_t height() const { return m_height; }

    __host__ __device__ Vec3 at(std::size_t row, std::size_t col) const {
        return m_pixels[row * width() + col];
    }
    __host__ __device__ Vec3& at(std::size_t row, std::size_t col) {
        return m_pixels[row * width() + col];
    }
};

// Writes a frame buffer of RGB floats into the given stream as PPM
inline std::ostream& operator<<(std::ostream& os, FrameBuffer& fb) {
    os << "P3\n" << fb.width() << ' ' << fb.height() << "\n255\n";

    for (std::size_t c { 0 }; c < fb.height(); ++c) {
        for (std::size_t r { 0 }; r < fb.width(); ++r) {
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

