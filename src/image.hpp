#pragma once

#include <cstdint>
#include <memory_resource>
#include <vector>

struct Rgb {
    float r, g, b;

    Rgb() = default;

    Rgb(float r, float g, float b)
        : r(r)
        , g(g)
        , b(b)
    {
    }

    static Rgb hex(uint8_t r, uint8_t g, uint8_t b);
};

class Image {
public:
    Image(size_t width, size_t height, std::pmr::memory_resource* res);

    Image(const Image& other, std::pmr::memory_resource* res);

    Image(const Image&& other);

    Image clone(std::pmr::memory_resource* res) const;

#if 0
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
#endif
    Image& operator=(const Image&);

    void set_pixel(size_t x, size_t y, Rgb color);
    Rgb get_pixel(size_t x, size_t y) const;

    void blend_pixel(const Image& target, size_t x, size_t y, Rgb color, float alpha, bool is_update_dist);

    float dist(const Image& target) const;

    float get_dist() const {
        return m_cached_dist;
    }

    void set_dist(float dist) {
        m_cached_dist = dist;
    };

    size_t width() const
    {
        return m_width;
    }

    size_t height() const
    {
        return m_height;
    }

private:
    std::pmr::vector<float> m_storage{};
    float* m_data{};
    size_t m_data_size{};
    size_t m_width{};
    size_t m_height{};
    float m_cached_dist{};
};
