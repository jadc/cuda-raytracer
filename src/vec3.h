#pragma once

class Vec3 {
    float m_components[3];
public:
    Vec3() : m_components{} {}
    Vec3(float x, float y, float z) : m_components{x, y, z} {}

    // Getters for each component
    float x() const { return m_components[0]; }
    float y() const { return m_components[1]; }
    float z() const { return m_components[2]; }

    // Overloads to allow indexing (const is copied, non-const is referenced)
    float operator[](int i) const { return m_components[i]; }
    float& operator[](int i) { return m_components[i]; }
};
