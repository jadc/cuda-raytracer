#pragma once

#include <cuda_runtime.h>
#include <cuda/std/optional>
#include <curand_kernel.h>
#include <cstdint>
#include <utility>

#include "cuda.h"
#include "math.h"

struct Scatter {
    Vec3 attenuation;
    Ray scattered;
};

#define MATERIAL_LIST(X) \
    X(Lambertian) \
    X(Metal)

#define MATERIAL_ENUM(Type) Type,
enum class MaterialType : uint8_t { MATERIAL_LIST(MATERIAL_ENUM) };
#undef MATERIAL_ENUM



/***** Materials *****/
struct Lambertian {
    static constexpr MaterialType tag = MaterialType::Lambertian;

    Vec3 albedo;

    __device__ cuda::std::optional<Scatter> scatter(const Ray& ray, const Hit& hit, curandState* rng) const {
        auto direction { hit.normal + Vec3::random_unit_vector(rng) };

        // Fallback to original normal if scattered vec is degenerate
        if (direction.near_zero())
            direction = hit.normal;

        return Scatter{ .attenuation = albedo, .scattered = Ray{hit.point, direction} };
    }
};

struct Metal {
    static constexpr MaterialType tag = MaterialType::Metal;

    Vec3 albedo;

    __device__ cuda::std::optional<Scatter> scatter(const Ray& ray, const Hit& hit, curandState* rng) const {
        const auto reflected { Vec3::reflect(ray.direction(), hit.normal) };
        return Scatter{ .attenuation = albedo, .scattered = Ray{hit.point, reflected} };
    }
};
/********************/



struct Material {
    MaterialType type;
    union Data {
        #define VARIANT(Type) Type Type;
        MATERIAL_LIST(VARIANT)
        #undef VARIANT
    } data;

    __device__ cuda::std::optional<Scatter> scatter(const Ray& ray, const Hit& hit, curandState* rng) const {
        switch (type) {
            #define MATCH(Type) case MaterialType::Type: return data.Type.scatter(ray, hit, rng);
            MATERIAL_LIST(MATCH)
            #undef MATCH
        }
        return cuda::std::nullopt;
    }
};

class MaterialTable {
    std::size_t m_count;
    const std::size_t m_capacity;
    UnifiedMemory<Material> m_materials;
public:
    __host__ MaterialTable(std::size_t capacity)
        : m_count{0}
        , m_capacity{capacity}
        , m_materials{capacity} {}

    template <typename Mat, typename... Args>
    __host__ Material* emplace(Args&&... args) {
        assert(m_count < m_capacity && "Material table capacity is full");
        auto& obj = m_materials[m_count++];
        obj.type = Mat::tag;
        new (&obj.data) Mat{std::forward<Args>(args)...};
        return &obj;
    }
};
