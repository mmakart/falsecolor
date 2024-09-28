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

    Image clone(std::pmr::memory_resource* res) const;

    Image(Image&& other)
    {
        m_width = other.m_width;
        m_height = other.m_height;
        m_data_size = other.m_data_size;
        m_data = other.m_data;
        m_cached_dist = other.m_cached_dist;
        m_storage = std::move(other.m_storage);
    }

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    void set_pixel(size_t x, size_t y, Rgb color);
    Rgb get_pixel(size_t x, size_t y) const;

    void blend_pixel(const Image& target, size_t x, size_t y, Rgb color, float alpha, bool is_update_dist);

    float dist(const Image& target) const;

    float dist_diff(const Image& target, size_t x, size_t y, Rgb color, float alpha) const;

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
