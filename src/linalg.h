#pragma once

#include <cuda_runtime.h>
#include <array>
#include <ostream>

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

    // Unary overloads
    __host__ __device__ float operator[](std::size_t i) const { return m_components[i]; }
    __host__ __device__ float& operator[](std::size_t i) { return m_components[i]; }

    // Print vector
    friend std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
        return os << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")";
    }

    // Vec3 * Scalar (explicitly commutative)
    __host__ __device__ Vec3 operator*(float scale) const {
        return { x() * scale, y() * scale, z() * scale };
    }
    __host__ __device__ friend Vec3 operator*(float scale, const Vec3& vec) {
        return vec * scale;
    }

    // Vec3 on Vec3 operations (implicitly commutative)
    __host__ __device__ Vec3 operator+(const Vec3& other) const {
        return { x() + other.x(), y() + other.y(), z() + other.z() };
    }
    __host__ __device__ Vec3 operator-(const Vec3& other) const {
        return { x() - other.x(), y() - other.y(), z() - other.z() };
    }
    __host__ __device__ Vec3 operator*(const Vec3& other) const {
        return { x() * other.x(), y() * other.y(), z() * other.z() };
    }
};

class Ray {
    Vec3 m_origin;
    Vec3 m_direction;
public:
    __device__ Ray()
        : m_origin{}
        , m_direction{}
        {}

    __device__ Ray(Vec3&& origin, Vec3&& direction)
        : m_origin{origin}
        , m_direction{direction}
        {}

    __device__ const Vec3& origin() const { return m_origin; }
    __device__ const Vec3& direction() const { return m_direction; }

    // A vector on the ray with magnitude t.
    __device__ Vec3 at(float t) {
        return origin() + t * direction();
    }
};
