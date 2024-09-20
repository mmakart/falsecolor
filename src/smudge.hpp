#pragma once

#include "image.hpp"
#include <string>

struct Brush {
    Rgb color;
    std::string name;

    Brush(std::string name, Rgb color)
        : color(color)
        , name(name)
    {
    }
};

struct Smudge {
    size_t x{};
    size_t y{};
    size_t brush_index{};

    Smudge() = default;

    Smudge make_variation(size_t width, size_t height, size_t brush_count) const;
    void apply(Image& image, const Image& target, const std::vector<Brush>& brushes) const;
};

struct SmudgePattern {
    Smudge smudge1{};
    Smudge smudge2{};
    bool has_smudge2{};

    SmudgePattern() = default;

    static SmudgePattern make_random(size_t width, size_t height, size_t brush_count);

    SmudgePattern make_variation(size_t width, size_t height, size_t brush_count) const;
};
