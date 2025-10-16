#include <curand_kernel.h>

__global__ void init_curand_states(curandState *states, unsigned long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}