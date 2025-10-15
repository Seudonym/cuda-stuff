#include "image.hpp"
#include <fstream>
#include <stdexcept>
#include <limits>

void write_pgm(uint32_t *array, size_t width, size_t height, const char *filename)
{
    // Find min/max value for normalization
    uint32_t max_val = 0, min_value = std::numeric_limits<uint32_t>::max();
    for (size_t i = 0; i < width * height; ++i)
    {
        if (array[i] > max_val)
            max_val = array[i];
        if (array[i] < min_value)
            min_value = array[i];
    }

    uint32_t range = max_val - min_value;
    if (range == 0)
        range = 1;

    // Open file for binary writing
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file for writing");
    }

    // Write PGM header and pixel data
    file << "P5\n"
         << width << " " << height << "\n255\n";
    for (size_t i = 0; i < width * height; ++i)
    {
        float normalized = static_cast<float>(array[i] - min_value) / range;
        uint8_t pixel = static_cast<uint8_t>(normalized * 255);
        file.write(reinterpret_cast<char *>(&pixel), sizeof(uint8_t));
    }
    file.close();
}

void write_ppm(uint32_t *red_array, uint32_t *green_array, uint32_t *blue_array,
               size_t width, size_t height, const char *filename)
{
    // Find min/max for each channel independently
    uint32_t max_r = 0, min_r = std::numeric_limits<uint32_t>::max();
    uint32_t max_g = 0, min_g = std::numeric_limits<uint32_t>::max();
    uint32_t max_b = 0, min_b = std::numeric_limits<uint32_t>::max();

    for (size_t i = 0; i < width * height; ++i)
    {
        if (red_array[i] > max_r)
            max_r = red_array[i];
        if (red_array[i] < min_r)
            min_r = red_array[i];

        if (green_array[i] > max_g)
            max_g = green_array[i];
        if (green_array[i] < min_g)
            min_g = green_array[i];

        if (blue_array[i] > max_b)
            max_b = blue_array[i];
        if (blue_array[i] < min_b)
            min_b = blue_array[i];
    }

    uint32_t range_r = max_r;
    uint32_t range_g = max_g;
    uint32_t range_b = max_b;

    if (range_r == 0)
        range_r = 1;
    if (range_g == 0)
        range_g = 1;
    if (range_b == 0)
        range_b = 1;

    // Open file for binary writing
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file for writing");
    }

    // Write PPM header (P6 for binary format)
    file << "P6\n"
         << width << " " << height << "\n255\n";

    // Write RGB pixel data (3 bytes per pixel)
    for (size_t i = 0; i < width * height; ++i)
    {
        // Normalize each channel independently
        float norm_r = static_cast<float>(red_array[i]) / range_r;
        float norm_g = static_cast<float>(green_array[i]) / range_g;
        float norm_b = static_cast<float>(blue_array[i]) / range_b;

        uint8_t pixel_r = static_cast<uint8_t>(norm_r * 255);
        uint8_t pixel_g = static_cast<uint8_t>(norm_g * 255);
        uint8_t pixel_b = static_cast<uint8_t>(norm_b * 255);

        // Write RGB triplet
        file.write(reinterpret_cast<char *>(&pixel_r), sizeof(uint8_t));
        file.write(reinterpret_cast<char *>(&pixel_g), sizeof(uint8_t));
        file.write(reinterpret_cast<char *>(&pixel_b), sizeof(uint8_t));
    }
    file.close();
}
