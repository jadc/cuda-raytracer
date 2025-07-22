#pragma once

#include <cuda_runtime.h>

#include "linalg.h"
#include "hittable.h"

class Sphere : Hittable {
    Vec3 m_center;
    float m_radius;
public:
    __device__ Sphere(Vec3 center, float radius)
        : m_center{std::move(center)}
        , m_radius{std::fmax(radius, 0.0f)}
        {};

    __device__ bool hit(const Ray& ray, float t_min, float t_max, Hit& hit) const override {
        const Vec3 oc { m_center - ray.origin() };
        const auto a { ray.direction().length_squared() };
        const auto h { Vec3::dot(ray.direction(), oc) };
        const auto c { oc.length_squared() - m_radius * m_radius };

        const auto discriminant { h * h - a * c };
        if (discriminant < 0) return false;

        // Nearest root within the interval [t_min, t_max]
        const auto sqrtd = std::sqrt(discriminant);
        auto root { (h - sqrtd) / a };
        if (root <= t_min || t_max <= root) {
            root = (h + sqrtd) / a;
            if (root <= t_min || t_max <= root)
                return false;
        }

        hit.t = root;
        hit.point = ray.at(hit.t);
        hit.set_face_normal(ray, /*outward_normal=*/{ (hit.point - m_center) / m_radius });

        return true;
    }
};
