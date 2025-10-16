#pragma once
#include <curand_kernel.h>

__global__ void init_curand_states(curandState *states, unsigned long seed);