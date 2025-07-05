#pragma once

#include <cuda_runtime.h>
#include <array>

class Vec3 {
    float m_components[3];
public:
    __host__ __device__ Vec3()
        : m_components{}
        {}

    __host__ __device__ Vec3(float x, float y, float z)
        : m_components{x, y, z}
        {}

    // Getters for each component
    __host__ __device__ float x() const { return m_components[0]; }
    __host__ __device__ float y() const { return m_components[1]; }
    __host__ __device__ float z() const { return m_components[2]; }

    // Overloads to allow indexing (const is copied, non-const is referenced)
    float operator[](std::size_t i) const { return m_components[i]; }
    float& operator[](std::size_t i) { return m_components[i]; }
};
