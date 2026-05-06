#pragma once

#include <cuda_runtime.h>
#include <utility>
#include <cassert>

#include "cuda.h"
#include "shapes.h"

class World {
    std::size_t m_count;
    const std::size_t m_capacity;
    UnifiedMemory<Hittable> m_objects;
public:
    __host__ World(std::size_t capacity)
        : m_count{0}
        , m_capacity{capacity}
        , m_objects{capacity}
        {};

    // Construct new Shape in unified memory in-place
    template <typename Shape, typename... Args>
    __host__ void emplace(Args&&... args) {
        assert(m_count < m_capacity && "World capacity is full");
        auto& obj = m_objects[m_count++];
        obj.type = Shape::tag;
        new (&obj.data) Shape{std::forward<Args>(args)...};
    }

    __device__ cuda::std::optional<Hit> hit(const Ray& ray, float t_min, float t_max) const {
        cuda::std::optional<Hit> hit;
        auto closest_so_far { t_max };

        // If the ray cast collides with any objects in world, return the closest
        for (std::size_t i { 0 }; i < m_count; ++i) {
            if (const auto obj { m_objects[i].hit(ray, t_min, closest_so_far) }) {
                closest_so_far = obj->t;
                hit = obj;
            }
        }

        return hit;
    }
};
