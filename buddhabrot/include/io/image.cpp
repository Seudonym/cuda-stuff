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