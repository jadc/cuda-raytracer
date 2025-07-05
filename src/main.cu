#include <fstream>
#include <cuda_runtime.h>

#include "vec3.h"
#include "framebuffer.h"

__global__ void hello_from_gpu() {
    printf("GPU thread %d\n", threadIdx.x);
}

constexpr int width { 256 };
constexpr int height { 256 };

int main() {
    FrameBuffer<width, height> fb {};
    for (int c { 0 }; c < width; ++c) {
        for (int r { 0 }; r < height; ++r) {
            const Vec3 pixel {
                float(r) / (width-1),
                float(c) / (height-1),
                0.0,
            };

            fb.at(r, c) = std::move(pixel);
        }
    }

    // Write frame buffer to ppm
    std::ofstream file { "output.ppm" };
    file << fb;
    file.close();  // RAII later
    return 0;
}
