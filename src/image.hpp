#pragma once

#include <cstdint>
#include <memory_resource>
#include <vector>

template <typename DType> // float or double
struct Rgb {
    DType r{};
    DType g{};
    DType b{};

    static constexpr Rgb<DType> hex(uint8_t r8, uint8_t g8, uint8_t b8);

    Rgb<DType>& operator+=(const Rgb<DType>& other) {
        r += other.r;
        g += other.g;
        b += other.b;

        return *this;
    }

    friend Rgb<DType> operator+(const Rgb<DType>& c1, const Rgb<DType>& c2) {
        return {c1.r + c2.r, c1.g + c2.g, c1.b + c2.b};
    }
};

template <typename DType> // float or double
constexpr Rgb<DType> Rgb<DType>::hex(uint8_t r8, uint8_t g8, uint8_t b8)
{
    return Rgb<DType>{
            static_cast<DType>(r8 / 255.0),
            static_cast<DType>(g8 / 255.0),
            static_cast<DType>(b8 / 255.0),
    };
}

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
        m_storage = std::move(other.m_storage);
    }

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    template <typename DType>
    void set_pixel(size_t x, size_t y, Rgb<DType> color)
    {
        if (x >= m_width || y >= m_height) {
            return;
        }

        size_t offset = 3 * (m_width * y + x);

        m_data[offset] = static_cast<float>(color.r);
        m_data[offset + 1] = static_cast<float>(color.g);
        m_data[offset + 2] = static_cast<float>(color.b);
    }

    template <typename DType>
    Rgb<DType> get_pixel(size_t x, size_t y) const
    {
        if (x >= m_width || y >= m_height) {
            return Rgb<DType>::hex(255, 255, 255);
        }

        size_t offset = 3 * (m_width * y + x);

        Rgb<DType> color;

        color.r = static_cast<DType>(m_data[offset]);
        color.g = static_cast<DType>(m_data[offset + 1]);
        color.b = static_cast<DType>(m_data[offset + 2]);

        return color;
    }

    template <typename DType>
    void blend_pixel(size_t x, size_t y, Rgb<DType> color, float alpha)
    {
        if (x >= m_width || y >= m_height) {
            return;
        }

        Rgb<DType> dst = get_pixel<DType>(x, y);

        float r = dst.r * (1.0f - alpha) + static_cast<float>(color.r) * alpha;
        float g = dst.g * (1.0f - alpha) + static_cast<float>(color.g) * alpha;
        float b = dst.b * (1.0f - alpha) + static_cast<float>(color.b) * alpha;

        set_pixel(x, y, Rgb<float>{r, g, b});
    }

    float dist(const Image& target) const;

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
};
