#pragma once
#include <cstddef>

class RenderConfig
{
public:
    /// @brief Construct a new RenderConfig object from a file
    /// @note The file should be in the following format:
    /// ```
    /// width: 800
    /// height: 600
    /// x_min: -2.5
    /// x_max: 1.5
    /// y_min: -2.0
    /// y_max: 2.0
    /// max_iterations: 2000
    /// samples_per_thread: 100000
    /// chunk_divisor: 10
    /// ```
    /// @param filename The name of the file to load the configuration from
    RenderConfig(const char *filename);

    /// @brief Print the configuration to standard output
    void print();

    size_t width, height;
    float x_min, x_max, y_min, y_max;
    size_t max_iterations, samples_per_thread, chunk_divisor;
};