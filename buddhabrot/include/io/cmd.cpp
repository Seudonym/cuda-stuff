#include "io/cmd.hpp"
#include "io/config.hpp"

#include <algorithm>
#include <iostream>

std::string get_cmd_option(char **begin, char **end, const std::string &option)
{
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return std::string(*itr);
    }
    return "";
}

bool cmd_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

void print_help()
{
    std::cerr << "Usage: buddhabrot [options]\n"
              << "Options:\n"
              << "  -h, --help              Show this help message and exit\n"
              << "  -s, --save-assets <dir> Save intermediate assets to files\n"
              << "  -i, --imp-map <file>    Use the specified importance map file (binary float array)\n"
              << "  -c, --config <file>     Specify the configuration file\n"
              << "  -o, --output <file>     Specify the output file\n";
}