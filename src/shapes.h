#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "linalg.h"

// Each enum should match a class/struct that implements `hit`
#define SHAPE_LIST(X) \
    X(Sphere)

#define SHAPE_ENUM(Type) Type,
enum class ShapeType : uint8_t { SHAPE_LIST(SHAPE_ENUM) };
#undef SHAPE_ENUM



/***** Shapes *****/
struct Sphere {
    static constexpr ShapeType tag = ShapeType::Sphere;

    Vec3 center;
    float radius;

    __host__ __device__ Sphere() : center{}, radius{0} {}

    __host__ __device__ Sphere(Vec3 center, float radius)
        : center{center}
        , radius{std::fmax(radius, 0.0f)}
        {}

    __device__ bool hit(const Ray& ray, float t_min, float t_max, Hit& hit) const {
        const Vec3 oc { center - ray.origin() };
        const auto a { ray.direction().length_squared() };
        const auto h { Vec3::dot(ray.direction(), oc) };
        const auto c { oc.length_squared() - radius * radius };

        const auto discriminant { h * h - a * c };
        if (discriminant < 0) return false;

        const auto sqrtd = std::sqrt(discriminant);
        auto root { (h - sqrtd) / a };
        if (root <= t_min || t_max <= root) {
            root = (h + sqrtd) / a;
            if (root <= t_min || t_max <= root)
                return false;
        }

        hit.t = root;
        hit.point = ray.at(hit.t);
        hit.set_face_normal(ray, (hit.point - center) / radius);

        return true;
    }
};
/******************/



struct Hittable {
    ShapeType type;
    union Data {
        #define VARIANT(Type) Type Type;
        SHAPE_LIST(VARIANT)
        #undef VARIANT
    } data;

    __device__ bool hit(const Ray& ray, float t_min, float t_max, Hit& rec) const {
        // Poor man's Rust enum matching, basically (CUDA doesn't support std::variant)
        switch (type) {
            #define MATCH(Type) case ShapeType::Type: return data.Type.hit(ray, t_min, t_max, rec);
            SHAPE_LIST(MATCH)
            #undef SHAPE_HIT_CASE
        }
        return false;
    }
};
