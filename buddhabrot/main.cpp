#include "include/buddhabrot.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  const size_t width = 2560;
  const size_t height = 2560;

  const float xmin = -2.5;
  const float xmax = 1.5;
  const float ymin = -2.0;
  const float ymax = 2.0;
  const size_t max_iter = 1000;
  const size_t samples_per_thread = 10000;

  uint32_t *histogram_g = new uint32_t[width * height]();
  generateBuddhabrot(histogram_g, width, height, xmin, xmax, ymin, ymax,
                     max_iter, samples_per_thread);

  // Save the histogram for later use
  std::ofstream hist_ofs("histogram.bin", std::ios::binary);
  hist_ofs.write(reinterpret_cast<char *>(histogram_g),
                 width * height * sizeof(uint32_t));
  hist_ofs.close();
  // Save the histogram as a PGM image
  std::ofstream ofs("buddhabrot.pgm", std::ios::binary);
  ofs << "P5\n" << width << " " << height << "\n255\n";
  for (size_t i = 0; i < width * height; ++i) {
    uint8_t pixel = static_cast<uint8_t>(
        std::min(histogram_g[i] / 256, static_cast<uint32_t>(255)));
    ofs.write(reinterpret_cast<char *>(&pixel), sizeof(uint8_t));
  }
  ofs.close();
  delete[] histogram_g;

  return 0;
}
