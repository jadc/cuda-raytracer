#pragma once

#include <ostream>
#include <array>
#include "vec3.h"

template <int m_width, int m_height>
class FrameBuffer {
    std::array<Vec3, m_width * m_height> m_pixels;
public:
    // Construct an empty frame buffer
    FrameBuffer() : m_pixels{} {}

    // Construct a frame buffer with n vectors
    template<typename... Args>
    FrameBuffer(Args... args) : m_pixels{args...} {}

    constexpr int width() const { return m_width; }
    constexpr int height() const { return m_height; }

    Vec3 at(int row, int col) const { return m_pixels[row * m_width + col]; }
    Vec3& at(int row, int col) { return m_pixels[row * m_width + col]; }
};

// Writes a frame buffer of RGB floats into the given stream as PPM
template <int m_width, int m_height>
inline std::ostream& operator<<(std::ostream& os, FrameBuffer<m_width, m_height>& fb) {
    os << "P3\n" << m_width << ' ' << m_height << "\n255\n";

    for (int c { 0 }; c < m_height; ++c) {
        for (int r { 0 }; r < m_width; ++r) {
            const Vec3& pixel { fb.at(r, c) };

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

