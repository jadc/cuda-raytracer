#pragma once

#include <optional>
#include <cuda_runtime.h>

#include "linalg.h"

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

// Interface for objects that a ray can hit
class Hittable {
public:
    // Virtual needed to call the destructor of derived classes as well
    virtual ~Hittable() = default;

    virtual bool hit(const Ray& ray, float t_min, float t_max, Hit& rec) const = 0;
};
