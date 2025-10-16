#include "io/config.hpp"
#include "kernels/kernels.hpp"
#include "curand_utils.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdint>

__global__ void importance_kernel(uint32_t *histogram, RenderConfig config,
                                  curandState *states)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state = states[thread_id];

    float sample_x_min = -2.1667f;
    float sample_x_max = +1.1667f;
    float sample_y_min = -1.667f;
    float sample_y_max = +1.667f;
    float sample_x_range = sample_x_max - sample_x_min;
    float sample_y_range = sample_y_max - sample_y_min;

    for (size_t sample = 0; sample < config.samples_per_thread; ++sample)
    {
        float cr = curand_uniform(&state) * sample_x_range + sample_x_min;
        float ci = curand_uniform(&state) * sample_y_range + sample_y_min;
        float zr = 0.0f, zi = 0.0f;
        float temp_zr = 0.0f;

        size_t iteration = 0, iterations_to_escape = 0;
        bool escaped = false;

        for (iteration = 0; iteration < config.max_iterations; ++iteration)
        {
            temp_zr = zr;
            zr = zr * zr - zi * zi + cr;
            zi = 2.0f * temp_zr * zi + ci;
            if (zr * zr + zi * zi > 8.0f)
            {
                escaped = true;
                iterations_to_escape = iteration;
                break;
            }
        }

        if (!escaped)
            continue;

        int pixel_x = static_cast<int>(((cr - sample_x_min) / sample_x_range) * config.width);
        int pixel_y = static_cast<int>(((ci - sample_y_min) / sample_y_range) * config.height);
        zr = 0.0f, zi = 0.0f;
        for (iteration = 0; iteration < iterations_to_escape; ++iteration)
        {
            temp_zr = zr;
            zr = zr * zr - zi * zi + cr;
            zi = 2.0f * temp_zr * zi + ci;

            if (zr <= config.x_max && zr >= config.x_min && zi <= config.y_max && zi >= config.y_min)
            {
                atomicAdd(&histogram[pixel_y * config.width + pixel_x], 1);
            }
        }
    }

    states[thread_id] = state;
}

void launch_importance_kernel(uint32_t *histogram, RenderConfig config)
{
    dim3 threads_per_block(256);
    dim3 num_blocks(1024);

    curandState *d_states;
    cudaMalloc(&d_states, threads_per_block.x * num_blocks.x * sizeof(curandState));
    init_curand_states<<<num_blocks, threads_per_block>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    uint32_t *d_histogram;
    cudaMalloc(&d_histogram, config.width * config.height * sizeof(uint32_t));
    cudaMemset(d_histogram, 0, config.width * config.height * sizeof(uint32_t));

    printf("Generating importance map...\n");

    importance_kernel<<<num_blocks, threads_per_block>>>(d_histogram, config, d_states);
    cudaDeviceSynchronize();

    printf("Done!\n");

    // Copy results back
    cudaMemcpy(histogram, d_histogram, config.width * config.height * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_histogram);
    cudaFree(d_states);
}