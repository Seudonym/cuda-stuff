#pragma once

#include <cstdint>
#include "io/config.hpp"

void launch_importance_kernel(uint32_t *histogram, RenderConfig config);
void launch_buddhabrot_kernel(uint32_t *histogram, RenderConfig config);
void launch_buddhabrot_rgb_kernel(uint32_t *r_hist, uint32_t *g_hist, uint32_t *b_hist, RenderConfig config);
