#pragma once
#include <cstdint>
#include <cstddef>

void write_pgm(uint32_t *array, size_t width, size_t height, const char *filename);
void write_ppm(uint32_t *red_array, uint32_t *green_array, uint32_t *blue_array,
               size_t width, size_t height, const char *filename);