#include "io/config.hpp"
#include "complex.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>

__global__ void buddhabrot_kernel(uint32_t *histogram, RenderConfig config)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    curandState_t state;
    curand_init(thread_id, 0, 0, &state);

    for (size_t sample = 0; sample < config.samples_per_thread; ++sample)
    {
        float x0 = curand_uniform(&state) * (config.x_max - config.x_min) + config.x_min;
        float y0 = curand_uniform(&state) * (config.y_max - config.y_min) + config.y_min;

        Complex c = Complex(x0, y0);
        Complex z = Complex(0.0f, 0.0f);

        size_t iteration = 0, iterations_to_escape = 0;
        bool escaped = false;

        // Iterate to see if the point escapes
        for (iteration = 0; iteration < config.max_iterations; ++iteration)
        {
            z = z * z + c;
            if (z.magnitude2() > 8.0f)
            {
                escaped = true;
                iterations_to_escape = iteration;
                break;
            }
        }

        // If the point did not escape, skip it
        if (!escaped)
            continue;

        z = Complex(0.0f, 0.0f);
        for (iteration = 0; iteration < iterations_to_escape; ++iteration)
        {
            z = z * z + c;

            int pixel_x = static_cast<int>(((z.real - config.x_min) / (config.x_max - config.x_min)) * config.width);
            int pixel_y = static_cast<int>(((z.imag - config.y_min) / (config.y_max - config.y_min)) * config.height);

            if (pixel_x >= 0 && pixel_x < static_cast<int>(config.width) &&
                pixel_y >= 0 && pixel_y < static_cast<int>(config.height))
            {
                atomicAdd(&histogram[pixel_y * config.width + pixel_x], 1);
            }
        }
    }
}

void launch_buddhabrot_kernel(uint32_t *histogram, RenderConfig config)
{
    dim3 threads_per_block(256);
    dim3 num_blocks(1024);

    uint32_t *d_histogram;
    cudaMalloc(&d_histogram, config.width * config.height * sizeof(uint32_t));
    cudaMemset(d_histogram, 0, config.width * config.height * sizeof(uint32_t));

    RenderConfig local_config = config;
    local_config.samples_per_thread /= local_config.chunk_divisor;

    // Progress bar
    printf("Rendering: 0%%");
    fflush(stdout);
    for (size_t chunk = 0; chunk < local_config.chunk_divisor; ++chunk)
    {
        buddhabrot_kernel<<<num_blocks, threads_per_block>>>(d_histogram, local_config);
        cudaDeviceSynchronize();
        printf("\rRendering: %zu%%", (chunk + 1) * 100 / local_config.chunk_divisor);
        fflush(stdout);
    }

    cudaMemcpy(histogram, d_histogram, config.width * config.height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_histogram);
}