#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <numbers>
#include <ostream>

namespace Math {
    constexpr float infinity { std::numeric_limits<float>::infinity() };
    constexpr float pi { std::numbers::pi_v<float> };

    inline float radians(float degrees) { return degrees * pi / 180.0f; }
}

class Vec3 {
public:
    float m_components[3] {};

    // Getters for each component
    __host__ __device__ float x() const { return m_components[0]; }
    __host__ __device__ float y() const { return m_components[1]; }
    __host__ __device__ float z() const { return m_components[2]; }

    // Unary overloads
    __host__ __device__ Vec3 operator-() const { return *this * -1; }
    __host__ __device__ float operator[](std::size_t i) const { return m_components[i]; }
    __host__ __device__ float& operator[](std::size_t i) { return m_components[i]; }

    // Print vector
    friend std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
        return os << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")";
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

    // Vec3 * Scalar (explicitly commutative)
    __host__ __device__ Vec3 operator*(float scale) const {
        return { x() * scale, y() * scale, z() * scale };
    }
    __host__ __device__ friend Vec3 operator*(float scale, const Vec3& vec) {
        return vec * scale;
    }

    // Vec3 / Scalar (explicitly commutative)
    __host__ __device__ Vec3 operator/(float scale) const {
        return { x() / scale, y() / scale, z() / scale };
    }
    __host__ __device__ friend Vec3 operator/(float scale, const Vec3& vec) {
        return vec / scale;
    }

    // Calculated properties

    // The length of the vector, squared
    __host__ __device__ float length_squared() const {
        return x() * x() + y() * y() + z() * z();
    }

    // The length of the vector
    __host__ __device__ float length() const {
        return std::sqrt(length_squared());
    }

    // A unit vector in the same direction as the given vector
    __host__ __device__ static Vec3 unit_vector(const Vec3& vec) {
        return vec / vec.length();
    }

    // The dot product between two vectors
    __host__ __device__ static float dot(const Vec3& a, const Vec3& b) {
        return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
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

    __device__ Ray(Vec3 origin, Vec3 direction)
        : m_origin{origin}
        , m_direction{direction}
        {}

    __device__ const Vec3& origin() const { return m_origin; }
    __device__ const Vec3& direction() const { return m_direction; }

    // A vector on the ray with magnitude t.
    __device__ Vec3 at(float t) const {
        return origin() + t * direction();
    }
};

struct Interval {
    // Default interval is empty
    float min { Math::infinity };
    float max { -1 * Math::infinity };

    __host__ __device__ constexpr bool size() const { return max - min; }
    __host__ __device__ constexpr bool contains(float val) const { return min <= val && val <= max; }
    __host__ __device__ constexpr bool surrounds(float val) const { return min < val && val < max; }
};

struct Hit {
    Vec3 point;
    Vec3 normal;
    float t;
    bool front_face;

    __device__ void set_face_normal(const Ray& ray, const Vec3& outward_normal) {
        front_face = Vec3::dot(ray.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};
