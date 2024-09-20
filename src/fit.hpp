#pragma once

#include "smudge.hpp"
#include "image.hpp"

std::vector<Smudge> fit_target_image(const Image& target, const std::vector<Brush>& brushes);
