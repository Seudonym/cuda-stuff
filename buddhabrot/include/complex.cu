#include "complex.cuh"
#include <cuda_runtime.h>

__host__ __device__ Complex::Complex(float r, float i) : real(r), imag(i) {}

__host__ __device__ float Complex::magnitude2() const
{
    return real * real + imag * imag;
}

__host__ __device__ Complex Complex::operator+(const Complex &other) const
{
    return Complex(real + other.real, imag + other.imag);
}

__host__ __device__ Complex Complex::operator*(const Complex &other) const
{
    return Complex(real * other.real - imag * other.imag,
                   real * other.imag + imag * other.real);
}

__host__ __device__ Complex Complex::operator*(float scalar) const
{
    return Complex(real * scalar, imag * scalar);
}

__host__ __device__ Complex Complex::operator-(const Complex &other) const
{
    return Complex(real - other.real, imag - other.imag);
}

__host__ __device__ Complex Complex::operator/(float scalar) const
{
    return Complex(real / scalar, imag / scalar);
}