#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

class Complex
{
public:
    float real;
    float imag;
    __host__ __device__ Complex(float r = 0.0f, float i = 0.0f);
    __host__ __device__ float magnitude2() const;
    __host__ __device__ Complex operator+(const Complex &other) const;
    __host__ __device__ Complex operator*(const Complex &other) const;
    __host__ __device__ Complex operator*(float scalar) const;
    __host__ __device__ Complex operator-(const Complex &other) const;
    __host__ __device__ Complex operator/(float scalar) const;
};