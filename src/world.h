#pragma once

#include <memory>
#include <vector>
#include <utility>

#include "cuda.h"
#include "linalg.h"
#include "hittable.h"

class World : Hittable {
    std::size_t m_num;
    Hittable** m_objects;
public:
    __host__ World(std::size_t num)
        : m_num{num}
    {
        /*
            Hittable **d_list;
        cuda_unwrap(cudaMalloc((void **)&d_list, 2*sizeof(Hittable *)));
   create_world<<<1,1>>>(d_list,d_world);
   cuda_unwrap(cudaGetLastError());
   cuda_unwrap(cudaDeviceSynchronize());
    */
    }

    __device__ ~World() {
        clear();
    }

    template <typename... Objs>
    __device__ void emplace(Objs&&... objs) {
        static_assert((std::is_same_v<std::decay_t<Objs>, Hittable> && ...),
            "All arguments must implement Hittable");
        sizeof...(objs);
    }

    __device__ void clear() {
        m_objects.clear();
    }

    __device__ bool hit(const Ray& ray, float t_min, float t_max, Hit& hit) const override {
        Hit temp_rec;
        bool hit_anything { false };
        auto closest_so_far { t_max };


        // Test if the ray cast collides with any objects in world
        // Update the temporary hit to the closest hit
        for (const auto& object : m_objects) {
            if (object->hit(ray, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                hit = temp_rec;
            }
        }

        return hit_anything;
    }
};
