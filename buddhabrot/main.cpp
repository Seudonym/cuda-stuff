#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

#include "io/cmd.hpp"
#include "io/config.hpp"
#include "io/image.hpp"
#include "kernels/kernels.hpp"

int main(int argc, char **argv)
{
  if (cmd_option_exists(argv, argv + argc, "-h") || cmd_option_exists(argv, argv + argc, "--help") || argc == 1)
  {
    print_help();
    return 0;
  }

  // Config file
  std::string config_file = get_cmd_option(argv, argv + argc, "-c");
  if (config_file.empty())
  {
    config_file = get_cmd_option(argv, argv + argc, "--config");
  }
  if (config_file.empty())
  {
    config_file = "render_config.txt";
  }

  // Output file
  std::string output_file = get_cmd_option(argv, argv + argc, "-o");
  if (output_file.empty())
  {
    output_file = get_cmd_option(argv, argv + argc, "--output");
  }
  if (output_file.empty())
  {
    output_file = "output.pgm";
  }

  // Importance map file
  std::string imp_map_file = get_cmd_option(argv, argv + argc, "-i");
  if (imp_map_file.empty())
  {
    imp_map_file = get_cmd_option(argv, argv + argc, "--imp-map");
  }
  if (imp_map_file.empty())
  {
    imp_map_file = "";
  }

  // Save assets directory
  std::string save_assets_dir = get_cmd_option(argv, argv + argc, "-s");
  if (save_assets_dir.empty())
  {
    save_assets_dir = get_cmd_option(argv, argv + argc, "--save-assets");
  }
  if (save_assets_dir.empty())
  {
    save_assets_dir = "";
  }

  // Print configuration
  std::cout << "Using output file: " << output_file << "\n";
  if (!imp_map_file.empty())
  {
    std::cout << "Using importance map file: " << imp_map_file << "\n";
  }
  if (!save_assets_dir.empty())
  {
    if (save_assets_dir.back() != '/' && save_assets_dir.back() != '\\')
    {
      save_assets_dir += '/';
    }
    std::cout << "Using directory to save assets: " << save_assets_dir << "\n";
    // Create directory if it doesn't exist
    std::string mkdir_cmd = "mkdir -p " + save_assets_dir;
    system(mkdir_cmd.c_str());
  }

  RenderConfig config(config_file.c_str());
  config.print();

  // uint32_t *histogram = new uint32_t[config.width * config.height]();
  // launch_buddhabrot_kernel(histogram, config);
  // write_pgm(histogram, config.width, config.height, output_file.c_str());

  uint32_t *r_hist = new uint32_t[config.width * config.height]();
  uint32_t *g_hist = new uint32_t[config.width * config.height]();
  uint32_t *b_hist = new uint32_t[config.width * config.height]();
  launch_buddhabrot_rgb_kernel(r_hist, g_hist, b_hist, config);
  write_ppm(r_hist, g_hist, b_hist, config.width, config.height, output_file.c_str());

  // if (!save_assets_dir.empty())
  // {
  //   std::string hist_file = save_assets_dir + "histogram.raw";
  //   std::ofstream hist_out(hist_file, std::ios::binary);
  //   hist_out.write(reinterpret_cast<char *>(histogram), config.width * config.height * sizeof(uint32_t));
  //   hist_out.close();
  //   std::cout << "Saved histogram to " << hist_file << "\n";
  // }

  return 0;
}