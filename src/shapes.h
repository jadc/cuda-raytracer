#pragma once

#include <cuda_runtime.h>
#include <cuda/std/optional>
#include <cstdint>

#include "math.h"

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

    __device__ cuda::std::optional<Hit> hit(const Ray& ray, Interval t) const {
        const auto oc { center - ray.origin() };
        const auto a { ray.direction().length_squared() };
        const auto h { Vec3::dot(ray.direction(), oc) };
        const auto c { oc.length_squared() - radius * radius };

        // If the ray hit the sphere, continue
        const auto discriminant { h * h - a * c };
        if (discriminant < 0) return cuda::std::nullopt;

        // Reject any hits outside the t interval
        const auto sqrtd = std::sqrt(discriminant);
        auto root { (h - sqrtd) / a };
        if (!t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!t.surrounds(root))
                return cuda::std::nullopt;
        }

        Hit hit {
            .point = ray.at(root),
            .t = root,
        };
        hit.set_face_normal(ray, (hit.point - center) / radius);

        return hit;
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

    __device__ cuda::std::optional<Hit> hit(const Ray& ray, Interval t) const {
        // Poor man's Rust enum matching, basically (CUDA doesn't support std::variant)
        switch (type) {
            #define MATCH(Type) case ShapeType::Type: return data.Type.hit(ray, t);
            SHAPE_LIST(MATCH)
            #undef SHAPE_HIT_CASE
        }
        return cuda::std::nullopt;
    }
};
