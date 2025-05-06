#include "image.hpp"
#include <cstring>
#include <memory_resource>

#define MAKE_ALIGNED_POINTER(pointer, alignment) \
    ((uintptr_t)(pointer) + (alignment - 1) & ~(uintptr_t)(alignment - 1))

Rgb Rgb::hex(uint8_t r, uint8_t g, uint8_t b)
{
    Rgb color;

    color.r = float(r) / 255.0f;
    color.g = float(g) / 255.0f;
    color.b = float(b) / 255.0f;

    return color;
}

Rgb blend_pixels(const Rgb& base, const Rgb& c, float alpha)
{
    float r = base.r * (1.0f - alpha) + c.r * alpha;
    float g = base.g * (1.0f - alpha) + c.g * alpha;
    float b = base.b * (1.0f - alpha) + c.b * alpha;
    return Rgb(r, g, b);
}

Image::Image(size_t width, size_t height, std::pmr::memory_resource* res)
    : m_width(width)
    , m_height(height)
    , m_data_size(width * height * 3)
    , m_storage(width * height * 3 + 32, 1.0f, res)
    , m_cached_dist(-1.0f) // default value means not yet calculated distance
{
    m_data = (float*)MAKE_ALIGNED_POINTER(m_storage.data(), 32);
}

Image Image::clone(std::pmr::memory_resource* res) const
{
    Image im(m_width, m_height, res);
    memcpy(im.m_data, m_data, m_data_size * sizeof(float));

    return im;
};

void Image::set_pixel(size_t x, size_t y, Rgb color)
{
    if (x >= m_width || y >= m_height) {
        return;
    }

    size_t offset = 3 * (m_width * y + x);

    m_data[offset] = color.r;
    m_data[offset + 1] = color.g;
    m_data[offset + 2] = color.b;
}

Rgb Image::get_pixel(size_t x, size_t y) const
{
    if (x >= m_width || y >= m_height) {
        return Rgb{255, 255, 255};
    }

    size_t offset = 3 * (m_width * y + x);

    Rgb color;

    color.r = m_data[offset];
    color.g = m_data[offset + 1];
    color.b = m_data[offset + 2];

    return color;
}

void Image::blend_pixel(const Image& target, size_t x, size_t y, Rgb color, float alpha, bool is_update_dist = true)
{
    if (x < 0 || y < 0 || x >= m_width || y >= m_height) {
        return;
    }

    Rgb dst = get_pixel(x, y);

    float r = dst.r * (1.0f - alpha) + color.r * alpha;
    float g = dst.g * (1.0f - alpha) + color.g * alpha;
    float b = dst.b * (1.0f - alpha) + color.b * alpha;

    set_pixel(x, y, Rgb(r, g, b));

    if (is_update_dist) {
        m_cached_dist = dist(target);
    }
}

float Image::dist(const Image& target) const
{
    float out{0.0f};

    float avg_r{};
    for (size_t i = 0; i < m_data_size; i += 3) {
        float r1{m_data[i]};
        float g1{m_data[i + 1]};
        float b1{m_data[i + 2]};
        float r2{target.m_data[i]};
        float g2{target.m_data[i + 1]};
        float b2{target.m_data[i + 2]};

        float avg_r{0.5f * (r1 + r2)};
        float dr{r1 - r2};
        float dg{g1 - g2};
        float db{b1 - b2};
        float dr_sqr{dr * dr};
        float dg_sqr{dg * dg};
        float db_sqr{db * db};
        float coef_r{2.0f + avg_r};
        float coef_g{4.0f};
        float coef_b{2.0f - avg_r};

        float dist_r{coef_r * dr_sqr};
        float dist_g{coef_g * dg_sqr};
        float dist_b{coef_b * db_sqr};

        out += dist_r * dist_r;
        out += dist_g * dist_g;
        out += dist_b * dist_b;
    }
    return out;
}

