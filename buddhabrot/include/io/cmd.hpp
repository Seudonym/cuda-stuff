#pragma once
#include <string>

std::string get_cmd_option(char **begin, char **end, const std::string &option);
bool cmd_option_exists(char **begin, char **end, const std::string &option);
void print_help();