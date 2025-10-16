#include "io/config.hpp"
#include "kernels/kernels.hpp"
#include "curand_utils.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdint>

// Add error checking macro
#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

__global__ void buddhabrot_kernel(uint32_t *histogram, RenderConfig config,
                                  curandState *states)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state = states[thread_id];

    float x_range = config.x_max - config.x_min;
    float y_range = config.y_max - config.y_min;

    for (size_t sample = 0; sample < config.samples_per_thread; ++sample)
    {
        float cr = curand_uniform(&state) * x_range + config.x_min;
        float ci = curand_uniform(&state) * y_range + config.y_min;
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

        zr = 0.0f, zi = 0.0f;
        for (iteration = 0; iteration < iterations_to_escape; ++iteration)
        {
            temp_zr = zr;
            zr = zr * zr - zi * zi + cr;
            zi = 2.0f * temp_zr * zi + ci;

            int pixel_x = static_cast<int>(((zr - config.x_min) / x_range) * config.width);
            int pixel_y = static_cast<int>(((zi - config.y_min) / y_range) * config.height);

            if (pixel_x >= 0 && pixel_x < static_cast<int>(config.width) &&
                pixel_y >= 0 && pixel_y < static_cast<int>(config.height))
            {
                atomicAdd(&histogram[pixel_y * config.width + pixel_x], 1);
            }
        }
    }

    states[thread_id] = state;
}

__global__ void buddhabrot_rgb_kernel(uint32_t *r_hist, uint32_t *g_hist, uint32_t *b_hist, RenderConfig config,
                                      curandState *states)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state = states[thread_id];

    float x_range = config.x_max - config.x_min;
    float y_range = config.y_max - config.y_min;

    for (size_t sample = 0; sample < config.samples_per_thread; ++sample)
    {
        float cr = curand_uniform(&state) * x_range + config.x_min;
        float ci = curand_uniform(&state) * y_range + config.y_min;
        float zr = 0.0f, zi = 0.0f;
        float temp_zr = 0.0f;

        size_t iteration = 0, iterations_to_escape = 0;
        bool escaped = false;

        for (iteration = 0; iteration < config.max_iterations; ++iteration)
        {
            temp_zr = zr;
            zr = zr * zr - zi * zi + cr;
            zi = 2.0f * temp_zr * zi + ci;
            iterations_to_escape++;
            if (zr * zr + zi * zi > 4.0f)
            {
                escaped = true;
                break;
            }
        }

        if (!escaped)
            continue;

        zr = 0.0f, zi = 0.0f;
        for (iteration = 0; iteration < iterations_to_escape; ++iteration)
        {
            temp_zr = zr;
            zr = zr * zr - zi * zi + cr;
            zi = 2.0f * temp_zr * zi + ci;

            int pixel_x = static_cast<int>(((zr - config.x_min) / x_range) * config.width);
            int pixel_y = static_cast<int>(((zi - config.y_min) / y_range) * config.height);

            if (pixel_x >= 0 && pixel_x < static_cast<int>(config.width) &&
                pixel_y >= 0 && pixel_y < static_cast<int>(config.height))
            {
                if (iterations_to_escape < config.b_thresh)
                    atomicAdd(&b_hist[pixel_y * config.width + pixel_x], 1);
                if (iterations_to_escape < config.g_thresh)
                    atomicAdd(&g_hist[pixel_y * config.width + pixel_x], 1);
                if (iterations_to_escape < config.r_thresh)
                    atomicAdd(&r_hist[pixel_y * config.width + pixel_x], 1);
            }
        }
    }
    states[thread_id] = state;
}

void launch_buddhabrot_kernel(uint32_t *histogram, RenderConfig config)
{
    dim3 threads_per_block(256);
    dim3 num_blocks(1024);

    curandState *d_states;
    cudaMalloc(&d_states, threads_per_block.x * num_blocks.x * sizeof(curandState));
    cudaCheckErrors("cudaMalloc states error");
    init_curand_states<<<num_blocks, threads_per_block>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    cudaCheckErrors("init curand error");

    uint32_t *d_histogram;
    cudaMalloc(&d_histogram, config.width * config.height * sizeof(uint32_t));
    cudaCheckErrors("cudaMalloc histogram error");
    cudaMemset(d_histogram, 0, config.width * config.height * sizeof(uint32_t));
    cudaCheckErrors("cudaMemset error");

    printf("Rendering buddhabrot grayscale...\n");

    RenderConfig local_config = config;
    local_config.samples_per_thread = config.samples_per_thread / config.num_chunks;

    for (size_t c = 0; c < config.num_chunks; c++)
    {
        buddhabrot_kernel<<<num_blocks, threads_per_block>>>(d_histogram, local_config, d_states);
        cudaDeviceSynchronize();
        printf("  Chunk: (%d/%d)\n", c + 1, config.num_chunks);
    }
    printf("\n");
    printf("Done!\n");

    // Copy results back
    cudaMemcpy(histogram, d_histogram, config.width * config.height * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy histogram error");

    // Cleanup
    cudaFree(d_histogram);
    cudaFree(d_states);
}

void launch_buddhabrot_rgb_kernel(uint32_t *r_hist, uint32_t *g_hist, uint32_t *b_hist, RenderConfig config)
{
    dim3 threads_per_block(256);
    dim3 num_blocks(1024);

    curandState *d_states;
    cudaMalloc(&d_states, threads_per_block.x * num_blocks.x * sizeof(curandState));
    cudaCheckErrors("cudaMalloc states error");
    init_curand_states<<<num_blocks, threads_per_block>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    cudaCheckErrors("init curand error");

    uint32_t *d_r_hist, *d_g_hist, *d_b_hist;
    cudaMalloc(&d_r_hist, config.width * config.height * sizeof(uint32_t));
    cudaMalloc(&d_g_hist, config.width * config.height * sizeof(uint32_t));
    cudaMalloc(&d_b_hist, config.width * config.height * sizeof(uint32_t));
    cudaCheckErrors("cudaMalloc histogram error");
    cudaMemset(d_r_hist, 0, config.width * config.height * sizeof(uint32_t));
    cudaMemset(d_g_hist, 0, config.width * config.height * sizeof(uint32_t));
    cudaMemset(d_b_hist, 0, config.width * config.height * sizeof(uint32_t));
    cudaCheckErrors("cudaMemset error");

    printf("Rendering buddhabrot...\n");

    RenderConfig local_config = config;
    local_config.samples_per_thread = config.samples_per_thread / config.num_chunks;

    for (size_t c = 0; c < config.num_chunks; c++)
    {
        buddhabrot_rgb_kernel<<<num_blocks, threads_per_block>>>(d_r_hist, d_g_hist, d_b_hist, local_config, d_states);
        cudaDeviceSynchronize();
        printf("  Chunk: (%d/%d)\n", c + 1, config.num_chunks);
    }
    printf("\n");
    printf("Done!\n");

    // Copy results back
    cudaMemcpy(r_hist, d_r_hist, config.width * config.height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(g_hist, d_g_hist, config.width * config.height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_hist, d_b_hist, config.width * config.height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy histogram error");

    // Cleanup
    cudaFree(d_r_hist);
    cudaFree(d_g_hist);
    cudaFree(d_b_hist);
    cudaFree(d_states);
}