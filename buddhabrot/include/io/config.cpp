#include "config.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <iostream>

RenderConfig::RenderConfig(const char *filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open config file");
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, ':'))
        {
            std::string value_str;
            if (std::getline(iss, value_str))
            {
                value_str.erase(0, value_str.find_first_not_of(" \t"));
                if (key == "width")
                    this->width = std::stoul(value_str);
                else if (key == "height")
                    this->height = std::stoul(value_str);
                else if (key == "x_min")
                    this->x_min = std::stof(value_str);
                else if (key == "x_max")
                    this->x_max = std::stof(value_str);
                else if (key == "y_min")
                    this->y_min = std::stof(value_str);
                else if (key == "y_max")
                    this->y_max = std::stof(value_str);
                else if (key == "max_iterations")
                    this->max_iterations = std::stoul(value_str);
                else if (key == "samples_per_thread")
                    this->samples_per_thread = std::stoul(value_str);
                else if (key == "chunk_divisor")
                    this->chunk_divisor = std::stoul(value_str);
                else
                    std::cerr << "Warning: Unknown config key '" << key << "'\n";
            }
        }
    }
}

void RenderConfig::print()
{
    std::cout << "RenderConfig:\n"
              << "  width: " << width << "\n"
              << "  height: " << height << "\n"
              << "  x_min: " << x_min << "\n"
              << "  x_max: " << x_max << "\n"
              << "  y_min: " << y_min << "\n"
              << "  y_max: " << y_max << "\n"
              << "  max_iterations: " << max_iterations << "\n"
              << "  samples_per_thread: " << samples_per_thread << "\n"
              << "  chunk_divisor: " << chunk_divisor << "\n";
}