#pragma once
#include <cstdint>
#include <cuda_runtime.h>

__global__ void buddhabrotKernel(uint32_t *histogram_g, size_t width,
                                 size_t height, float xmin, float xmax,
                                 float ymin, float ymax, size_t max_iter,
                                 size_t samples_per_thread);
void generateBuddhabrot(uint32_t *histogram_g, size_t width, size_t height,
                        float xmin, float xmax, float ymin, float ymax,
                        size_t max_iter, size_t samples_per_thread);
