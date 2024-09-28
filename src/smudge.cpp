#include "smudge.hpp"
#include "random.hpp"
#include <algorithm>

Smudge Smudge::make_variation(size_t width, size_t height, size_t brush_count) const
{
    int xdelta = random<int>(-1, 1);
    int ydelta = random<int>(-1, 1);

    Smudge variation;

    variation.x = size_t(std::clamp(int(x) + xdelta, 0, int(width - 1)));
    variation.y = size_t(std::clamp(int(y) + ydelta, 0, int(height - 1)));

    variation.brush_index = brush_index;

    if (random<int>(0, 9) == 0) {
        variation.brush_index = random<size_t>(0, brush_count - 1);
    }

    return variation;
}

void Smudge::apply(Image& image, const Image& target, const std::vector<Brush>& brushes) const
{
    if (x < 0 || y < 0 || x >= image.width() || y >= image.height()) {
        return;
    }

    Rgb color = brushes[brush_index].color;

    image.blend_pixel(target, x, y, color, 0.7f, false);
    image.blend_pixel(target, x, y - 1, color, 0.5f, false);
    image.blend_pixel(target, x, y + 1, color, 0.5f, false);
    image.blend_pixel(target, x - 1, y, color, 0.5f, false);
    image.blend_pixel(target, x + 1, y, color, 0.5f, false);
    image.set_dist(image.dist(target));
}

SmudgePattern SmudgePattern::make_random(size_t width, size_t height, size_t brush_count)
{
    SmudgePattern pattern;
    pattern.smudge1.x = random<size_t>(0, width - 1);
    pattern.smudge1.y = random<size_t>(0, height - 1);
    pattern.smudge1.brush_index = random<size_t>(0, brush_count - 1);

    if (random<size_t>(0, 1)) {
        pattern.has_smudge2 = true;
        pattern.smudge2 = pattern.smudge1.make_variation(width, height, brush_count);
    }

    return pattern;
}

SmudgePattern SmudgePattern::make_variation(size_t width, size_t height, size_t brush_count) const
{
    SmudgePattern variation;
    variation.smudge1 = smudge1.make_variation(width, height, brush_count);
    variation.has_smudge2 = has_smudge2;
    if (has_smudge2) {
        variation.smudge2 = smudge2.make_variation(width, height, brush_count);
    }
    return variation;
}
