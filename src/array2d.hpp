#pragma once

#include "image.hpp"
#include "settings.hpp"
#include <array>
#include <vector>

template <typename T>
struct Array2D {
    using Container = std::vector<T>;
    using Iterator = typename Container::iterator;
    using ConstIterator = typename Container::const_iterator;

    Array2D(Signed width, Signed height)
        : m_width{width}
        , m_height{height}
        , m_data(width * height)
    {
    }

    Array2D(Signed width, Signed height, T init_value)
        : m_width{width}
        , m_height{height}
        , m_data(width * height, init_value)
    {
    }

    Array2D(const Image& im)
        : m_width{static_cast<Signed>(im.width())}
        , m_height{static_cast<Signed>(im.height())}
        , m_data(im.width() * im.height())
    {
        for (Signed y = 0; y < m_height; ++y) {
            for (Signed x = 0; x < m_width; ++x) {
                (*this)(x, y) = im.get_pixel<DType>(x, y);
            }
        }
    }

    T& operator()(Signed x, Signed y) {
        return m_data[static_cast<size_t>(y * m_width + x)];
    }

    const T& operator()(Signed x, Signed y) const {
        return m_data[static_cast<size_t>(y * m_width + x)];
    }

    Iterator begin() noexcept {
        return m_data.begin();
    }

    ConstIterator cbegin() const noexcept {
        return m_data.cbegin();
    }

    Iterator end() noexcept {
        return m_data.end();
    }

    ConstIterator cend() const noexcept {
        return m_data.cend();
    }

    Signed width() const {
        return m_width;
    }

    Signed height() const {
        return m_height;
    }

private:
    const Signed m_width;
    const Signed m_height;
    Container m_data;
};

template <typename T, size_t Width, size_t Height>
struct Array2DConstDim {
    using Container = std::array<T, Width * Height>;
    using Iterator = typename Container::iterator;
    using ConstIterator = typename Container::const_iterator;

    constexpr T& operator()(Signed x, Signed y) {
        return m_data[static_cast<size_t>(y * static_cast<Signed>(Width) + x)];
    }

    constexpr const T& operator()(Signed x, Signed y) const {
        return m_data[static_cast<size_t>(y * static_cast<Signed>(Width) + x)];
    }

    Iterator begin() noexcept {
        return m_data.begin();
    }

    ConstIterator cbegin() const noexcept {
        return m_data.cbegin();
    }

    Iterator end() noexcept {
        return m_data.end();
    }

    ConstIterator cend() const noexcept {
        return m_data.cend();
    }

    constexpr Signed width() const {
        return Width;
    }

    constexpr Signed height() const {
        return Height;
    }

private:
    Container m_data;
};

