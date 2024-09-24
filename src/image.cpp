#include "image.hpp"
#include <array>
#include <cstring>
#include <immintrin.h>
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


float color_dist(const Rgb& c1, const Rgb& c2)
{
    float avg_r = 0.5f * (c1.r + c2.r);
    float dr = c1.r - c2.r;
    float dg = c1.g - c2.g;
    float db = c1.b - c2.b;
    float dr_sqr = dr * dr;
    float dg_sqr = dg * dg;
    float db_sqr = db * db;
    float coef_r = 2.0f + avg_r;
    float coef_g = 4.0f;
    float coef_b = 2.0f - avg_r;
    float dist_r = coef_r * dr_sqr;
    float dist_g = coef_g * dg_sqr;
    float dist_b = coef_b * db_sqr;
    return dist_r * dist_r + dist_g * dist_g + dist_b * dist_b;
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

void Image::blend_pixel(const Image& target, size_t x, size_t y, Rgb color, float alpha)
{
    if (x >= m_width || y >= m_height) {
        return;
    }

    Rgb dst = get_pixel(x, y);

    float r = dst.r * (1.0f - alpha) + color.r * alpha;
    float g = dst.g * (1.0f - alpha) + color.g * alpha;
    float b = dst.b * (1.0f - alpha) + color.b * alpha;

    set_pixel(x, y, Rgb(r, g, b));

    m_cached_dist = dist(target);
}

#if 1
float Image::dist(const Image& target) const
{
    __m256 out = _mm256_setzero_ps();

    const __m256 const2 = _mm256_set1_ps(2.0f);
    const __m256 const4 = _mm256_set1_ps(4.0f);
    const __m256 const0_5 = _mm256_set1_ps(0.5f);

    __m256 avg_r, dr, dg, db, dr_sqr, dg_sqr, db_sqr, coef_r, coef_g, coef_b, dist_r, dist_g, dist_b;
    std::array<float, 8> r1_arr, g1_arr, b1_arr, r2_arr, g2_arr, b2_arr;
    // Assumption! Image width * height % 8 == 0
    for (size_t i = 0; i < m_data_size; i += 24) {
        for (size_t j = 0; j < 8; j++) {
            r1_arr[j] = m_data[i + 3 * j];
            g1_arr[j] = m_data[i + 3 * j + 1];
            b1_arr[j] = m_data[i + 3 * j + 2];
            r2_arr[j] = target.m_data[i + 3 * j];
            g2_arr[j] = target.m_data[i + 3 * j + 1];
            b2_arr[j] = target.m_data[i + 3 * j + 2];
        }

	__m256 r1 = _mm256_loadu_ps(r1_arr.data());
	__m256 g1 = _mm256_loadu_ps(g1_arr.data());
	__m256 b1 = _mm256_loadu_ps(b1_arr.data());
	__m256 r2 = _mm256_loadu_ps(r2_arr.data());
	__m256 g2 = _mm256_loadu_ps(g2_arr.data());
	__m256 b2 = _mm256_loadu_ps(b2_arr.data());

        avg_r = _mm256_mul_ps(const0_5, _mm256_add_ps(r1, r2));
        dr = _mm256_sub_ps(r1, r2);
        dg = _mm256_sub_ps(g1, g2);
        db = _mm256_sub_ps(b1, b2);
        dr_sqr = _mm256_mul_ps(dr, dr);
        dg_sqr = _mm256_mul_ps(dg, dg);
        db_sqr = _mm256_mul_ps(db, db);
        coef_r = _mm256_add_ps(const2, avg_r);
        coef_g = const4;
        coef_b = _mm256_sub_ps(const2, avg_r);

        dist_r = _mm256_mul_ps(coef_r, dr_sqr);
        dist_g = _mm256_mul_ps(coef_g, dg_sqr);
        dist_b = _mm256_mul_ps(coef_b, db_sqr);
        out = _mm256_add_ps(out, _mm256_mul_ps(dist_r, dist_r));
        out = _mm256_add_ps(out, _mm256_mul_ps(dist_g, dist_g));
        out = _mm256_add_ps(out, _mm256_mul_ps(dist_b, dist_b));
        // out = _mm256_add_ps(out, dist_r);
        // out = _mm256_add_ps(out, dist_g);
        // out = _mm256_add_ps(out, dist_b);
    }

    float ds = 0.0f;
    for (int i = 0; i < 8; i++) {
        ds += ((float*)&out)[i];
    }

    return ds;
}
#else
float Image::dist(const Image& other) const
{
    float d = 0.0f;

    for (size_t i = 0; i < m_data_size; i += 3) {
        float r1 = m_data[i];
        float g1 = m_data[i + 1];
        float b1 = m_data[i + 2];
        float r2 = other.m_data[i];
        float g2 = other.m_data[i + 1];
        float b2 = other.m_data[i + 2];
        float avg_r = (r1 + r2) / 2.0f;
        float dr = fabs(r1 - r2);
        float dg = fabs(g1 - g2);
        float db = fabs(b1 - b2);
        float coef_r = 2.0f + avg_r;
        float coef_g = 4.0f;
        float coef_b = 3.0f - avg_r;
        float delta_r = coef_r * dr;
        float delta_g = coef_g * dg;
        float delta_b = coef_b * db;

        d += r1 * r2 * pow(delta_r, 5) + g1 * g2 * pow(delta_g, 5) + b1 * b2 * pow(delta_b, 5);
    }

    return d;
}
#endif

float Image::dist_diff(const Image& target, size_t x, size_t y, Rgb color, float alpha) const
{
    if (x < 0 || y < 0 || x >= m_width || y >= m_height) {
        return 0.0f;
    }

    Rgb p1 = get_pixel(x, y);
    Rgb p2 = target.get_pixel(x, y);

    float prev_contribution = color_dist(p1, p2);
    Rgb p1_changed = blend_pixels(p1, color, alpha);
    float current_contribution = color_dist(p1_changed, p2);

    return current_contribution - prev_contribution;
}
