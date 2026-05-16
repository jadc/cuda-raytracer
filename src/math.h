#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <numbers>
#include <ostream>

namespace Math {
    constexpr float infinity { std::numeric_limits<float>::infinity() };
    constexpr float pi { std::numbers::pi_v<float> };

    __host__ __device__ constexpr float radians(float degrees) { return degrees * pi / 180.0f; }

    // A random float within the range [0.0f, 1.0f)
    __device__ static float random(curandState* state) {
        return 1.0f - curand_uniform(state);
    }

    // A random float within the specified range
    __device__ static float random(curandState* state, float min, float max) {
        return min + (max - min) * random(state);
    }
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

    // Compound assignment operators
    __host__ __device__ Vec3& operator+=(const Vec3& other) {
        m_components[0] += other.m_components[0];
        m_components[1] += other.m_components[1];
        m_components[2] += other.m_components[2];
        return *this;
    }
    __host__ __device__ Vec3& operator-=(const Vec3& other) {
        m_components[0] -= other.m_components[0];
        m_components[1] -= other.m_components[1];
        m_components[2] -= other.m_components[2];
        return *this;
    }
    __host__ __device__ Vec3& operator*=(const Vec3& other) {
        m_components[0] *= other.m_components[0];
        m_components[1] *= other.m_components[1];
        m_components[2] *= other.m_components[2];
        return *this;
    }
    __host__ __device__ Vec3& operator*=(float scale) {
        m_components[0] *= scale;
        m_components[1] *= scale;
        m_components[2] *= scale;
        return *this;
    }
    __host__ __device__ Vec3& operator/=(float scale) {
        m_components[0] /= scale;
        m_components[1] /= scale;
        m_components[2] /= scale;
        return *this;
    }

    // Vec3 on Vec3 operations
    __host__ __device__ Vec3 operator+(const Vec3& other) const { auto r = *this; r += other; return r; }
    __host__ __device__ Vec3 operator-(const Vec3& other) const { auto r = *this; r -= other; return r; }
    __host__ __device__ Vec3 operator*(const Vec3& other) const { auto r = *this; r *= other; return r; }

    // Vec3 * Scalar (commutative)
    __host__ __device__ Vec3 operator*(float scale) const { auto r = *this; r *= scale; return r; }
    __host__ __device__ friend Vec3 operator*(float scale, const Vec3& vec) { return vec * scale; }

    // Vec3 / Scalar (commutative)
    __host__ __device__ Vec3 operator/(float scale) const { auto r = *this; r /= scale; return r; }
    __host__ __device__ friend Vec3 operator/(float scale, const Vec3& vec) { return vec / scale; }

    // Calculated properties

    // The length of the vector, squared
    __host__ __device__ float length_squared() const {
        return x() * x() + y() * y() + z() * z();
    }

    // The length of the vector
    __host__ __device__ float length() const {
        return std::sqrt(length_squared());
    }

    // Whether or not this vector is near zero
    __host__ __device__ bool near_zero() const {
        constexpr float s { 1e-8f };
        return (std::fabs(x()) < s) && (std::fabs(y()) < s) && (std::fabs(z()) < s);
    }

    // A unit vector in the same direction as the given vector
    __host__ __device__ static Vec3 unit_vector(const Vec3& vec) {
        return vec / vec.length();
    }

    // The dot product between two vectors
    __host__ __device__ static float dot(const Vec3& a, const Vec3& b) {
        return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
    }

    // A random vector where each component is within the range [0.0f, 1.0f)
    __device__ static Vec3 random(curandState* rng) {
        return {
            Math::random(rng),
            Math::random(rng),
            Math::random(rng),
        };
    }

    // A random vector where each component is within the specified range
    __device__ static Vec3 random(curandState* rng, float min, float max) {
        return {
            Math::random(rng, min, max),
            Math::random(rng, min, max),
            Math::random(rng, min, max),
        };
    }

    // A unit vector in a random direction
    __device__ static Vec3 random_unit_vector(curandState* rng) {
        Vec3 v;
        float len;

        // Rejection-method: generate random vectors until one
        // is within a unit sphere (i.e. 0 < length^2 < 1)
        do {
            v = Vec3::random(rng, -1.0f, 1.0f);
            len = v.length_squared();
        } while (len <= 1e-12f || len > 1.0f);

        return v / std::sqrt(len);
    }

    // A unit vector in a random direction from a sphere hemisphere
    __device__ static Vec3 random_on_hemisphere(curandState* rng, const Vec3& normal) {
        const auto on_unit_sphere { random_unit_vector(rng) };
        return Vec3::dot(on_unit_sphere, normal) > 0.0f ? on_unit_sphere : -on_unit_sphere;
    }

    // A new vector which is v reflected on a surface with normal n
    __device__ static Vec3 reflect(const Vec3& v, const Vec3& n) {
        return v - 2 * Vec3::dot(v, n) * n;
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
    float max { -Math::infinity };

    __host__ __device__ constexpr float size() const {
        return max - min;
    }

    __host__ __device__ constexpr bool contains(float val) const {
        return min <= val && val <= max;
    }

    __host__ __device__ constexpr bool surrounds(float val) const {
        return min < val && val < max;
    }

    __host__ __device__ constexpr float clamp(float val) const {
        if (val < min) return min;
        if (val > max) return max;
        return val;
    }
};

// Forward declaration to avoid circular dependencies
struct Material;

struct Hit {
    Vec3 point;
    Vec3 normal;
    Material* mat;
    float t;
    bool front_face;

    __device__ void set_face_normal(const Ray& ray, const Vec3& outward_normal) {
        front_face = Vec3::dot(ray.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};
