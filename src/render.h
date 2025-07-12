#pragma once

#include <cuda_runtime.h>

#include "linalg.h"
#include "framebuffer.h"

__global__ void render(FrameBuffer* fb, const Vec3* camera_center, const Vec3* first_pixel, const Vec3* pixel_delta_u, const Vec3* pixel_delta_v);
