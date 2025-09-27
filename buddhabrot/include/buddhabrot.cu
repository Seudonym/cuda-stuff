#include "buddhabrot.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <curand_kernel.h>

__global__ void buddhabrotKernel(uint32_t *histogram_g, size_t width,
                                 size_t height, float xmin, float xmax,
                                 float ymin, float ymax, size_t max_iter,
                                 size_t samples_per_thread) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  curandState_t state;
  curand_init(1234ULL, tid, 0, &state);

  float xrange = xmax - xmin;
  float yrange = ymax - ymin;
  float range_multiplier = 3.0;

  for (size_t s = 0; s < samples_per_thread; s++) {
    // generate a random c point
    float cr = curand_uniform(&state) * xrange + xmin;
    cr *= range_multiplier;
    float ci = curand_uniform(&state) * yrange + ymin;
    ci *= range_multiplier;
    if (tid == 4 && s == 0) {
      printf("TID: %d, Sampled point: (%f, %f)\n", tid, cr, ci);
    }

    float zr = 0.0, zi = 0.0;
    // find trajectory
    size_t iter_to_escape = 0;
    bool escaped = false;

    for (size_t i = 0; i < max_iter; i++) {
      float new_zi = 2.0 * zr * zi + ci;
      float new_zr = zr * zr - zi * zi + cr;
      zr = new_zr;
      zi = new_zi;

      iter_to_escape++;
      if (zr * zr + zi * zi > 4.0) {
        if (tid == 4 && s == 0) {
          printf("TID: %d, Escaped at iteration: %llu\n", tid, iter_to_escape);
        }

        escaped = true;
        break;
      }
    }
    // if not escaping, then ignore sample
    if (!escaped) {
      if (tid == 4 && s == 0) {
        std::printf("TID: %d, did not escape after %llu iterations\n", tid,
                    max_iter);
      }
      continue;
    }

    if (tid == 4 && s == 0) {
      printf("TID: %d, Iterations to escape: %llu\n", tid, iter_to_escape);
    }

    // iterate across all stored trajectories and bump histogram bins
    zr = zi = 0.0;
    for (size_t i = 0; i < iter_to_escape; i++) {
      float new_zi = 2.0 * zr * zi + ci;
      float new_zr = zr * zr - zi * zi + cr;
      zr = new_zr;
      zi = new_zi;

      int x = (int)((zr - xmin) / xrange * width);
      int y = (int)((zi - ymin) / yrange * height);

      if (x >= 0 && x < width && y >= 0 && y < height) {
        atomicAdd(&histogram_g[x + y * width], 1);
      }
    }
  }
}

void generateBuddhabrot(uint32_t *histogram_g, size_t width, size_t height,
                        float xmin, float xmax, float ymin, float ymax,
                        size_t max_iter, size_t samples_per_thread) {
  size_t histogram_size_bytes = width * height * sizeof(uint32_t);
  uint32_t *d_histogram_g;
  cudaMalloc(&d_histogram_g, histogram_size_bytes);
  cudaMemset(d_histogram_g, 0, histogram_size_bytes);

  dim3 threads_per_block(256);
  dim3 blocks_per_grid(1024);
  buddhabrotKernel<<<blocks_per_grid, threads_per_block>>>(
      d_histogram_g, width, height, xmin, xmax, ymin, ymax, max_iter,
      samples_per_thread);
  cudaDeviceSynchronize();

  cudaMemcpy(histogram_g, d_histogram_g, histogram_size_bytes,
             cudaMemcpyDeviceToHost);
  cudaFree(d_histogram_g);
}
