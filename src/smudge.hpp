#pragma once

#include "image.hpp"
#include "settings.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>

template <typename DType>
struct Brush {
    std::string_view name;
    Rgb<DType> color;
};

template <typename DType> // float or double
struct SmudgeProperties {
    static constexpr size_t max_pixels{5};

    size_t num_pixels;
    std::array<Signed, max_pixels> xs;
    std::array<Signed, max_pixels> ys;
    std::array<DType, max_pixels> alphas;
    std::array<size_t, max_pixels> alphas_idxs;

    // Not std::string because it wouldn't allow constexpr initialization
    std::string_view type;
};

namespace PredefinedBrushes {
    enum BrushType : size_t { // For indexing
        pixel, water, oil, num_types
    };

    enum Alpha : size_t { // For indexing
        a0_5, a0_7, a0_9, a1_0, num_alphas
    };

    // Same order as in enum Alpha!
    inline constexpr std::array<DType, num_alphas> all_alphas {
            0.5, 0.7, 0.9, 1.0
    };

    inline constexpr std::array all_colors {
            Brush<DType>{"White", Rgb<DType>::hex(0xff, 0xff, 0xff)},
            Brush<DType>{"Yellow", Rgb<DType>::hex(0xff, 0xf0, 0x00)},
            Brush<DType>{"Orange", Rgb<DType>::hex(0xff, 0x6c, 0x00)},
            Brush<DType>{"Red", Rgb<DType>::hex(0xff, 0x00, 0x00)},
            Brush<DType>{"Violet", Rgb<DType>::hex(0x8a, 0x00, 0xff)},
            Brush<DType>{"Blue", Rgb<DType>::hex(0x00, 0x0c, 0xff)},
            Brush<DType>{"Green", Rgb<DType>::hex(0x0c, 0xff, 0x00)},
            Brush<DType>{"Magenta", Rgb<DType>::hex(0xfc, 0x00, 0xff)},
            Brush<DType>{"Cyan", Rgb<DType>::hex(0x00, 0xff, 0xea)},
            Brush<DType>{"Grey", Rgb<DType>::hex(0xbe, 0xbe, 0xbe)},
            Brush<DType>{"DarkGrey", Rgb<DType>::hex(0x7b, 0x7b, 0x7b)},
            Brush<DType>{"Black", Rgb<DType>::hex(0x00, 0x00, 0x00)},
            Brush<DType>{"DarkGreen", Rgb<DType>::hex(0x00, 0x64, 0x00)},
            Brush<DType>{"Brown", Rgb<DType>::hex(0x96, 0x4b, 0x00)},
            Brush<DType>{"Pink", Rgb<DType>::hex(0xff, 0xc0, 0xcb)},
    };

    inline constexpr size_t num_colors{std::size(all_colors)};

    inline constexpr SmudgeProperties<DType> pixel_props {
            1, {0}, {0}, {1.0}, {a1_0}, {"p"}
    };

    inline constexpr SmudgeProperties<DType> water_props {
            5,
            {0, -1, 0, 1, 0},
            {-1, 0, 0, 0, 1},
            {0.5, 0.5, 0.7, 0.5, 0.5},
            {a0_5, a0_5, a0_7, a0_5, a0_5},
            {"w"}
    };

    inline constexpr SmudgeProperties<DType> oil_props {
            5,
            {0, -1, 0, 1, 0},
            {-1, 0, 0, 0, 1},
            {0.9, 0.9, 1.0, 0.9, 0.9},
            {a0_9, a0_9, a1_0, a0_9, a0_9},
            {"o"}
    };

    inline constexpr Array2DConstDim<Rgb<DType>, num_alphas, num_colors> init_premul() {
        Array2DConstDim<Rgb<DType>, num_alphas, num_colors> result{};

        for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
            for (size_t alpha_idx = 0; alpha_idx < num_alphas; ++alpha_idx) {
                auto color = all_colors[color_idx].color;
                auto alpha = all_alphas[alpha_idx];

                result(alpha_idx, color_idx) = {
                        color.r * alpha, color.g * alpha, color.b * alpha
                };
            }
        }

        return result;
    }

    inline constexpr Array2DConstDim<Rgb<DType>, num_alphas, num_colors> premul{init_premul()};

    // Same order as in enum BrushType!
    inline constexpr std::array all_types{pixel_props, water_props, oil_props};
}

template <typename DType>
struct Smudge {
    Signed x{};
    Signed y{};
    size_t type_idx{};
    size_t color_idx{};

    void apply(Image& image) const
    {
        using PredefinedBrushes::all_types, PredefinedBrushes::all_colors;

        if (x < 0 || static_cast<size_t>(x) >= image.width() ||
                y < 0 || static_cast<size_t>(y) >= image.height()) {
            return;
        }

        const auto& props{all_types[type_idx]};
        const auto color{all_colors[color_idx].color};

        for (auto i = 0; i < props.num_pixels; ++i) {
            image.blend_pixel(
                    x + props.xs[i],
                    y + props.ys[i],
                    color,
                    props.alphas[i]
            );
        }
    }

    using SmudgePixelsData = std::vector<std::tuple<Signed, Signed, DType, size_t>>;

    // TODO: replace vector to avoid heap allocations
    constexpr SmudgePixelsData pixels_data() const {
        using PredefinedBrushes::all_types, PredefinedBrushes::num_types;

        const auto& props{all_types[type_idx]};

        SmudgePixelsData result(props.num_pixels);

        for (size_t i = 0; i < props.num_pixels; ++i) {
            result[i] = {
                    props.xs[i],
                    props.ys[i],
                    props.alphas[i],
                    props.alphas_idxs[i]
            };
        }

        return result;
    }
};

// Standard alpha blending formula: R = (1 - A) * D + A * S
// or R = (1 - A) * D + S_premul.
// Possible return value is in range [-1; 1].
// result and source_premul are expected to be in range [-1; 1].
// R = result, D = destination, S = source, A = alpha.
template <typename DType>
inline DType channel_error_from_reversed_blend(
        DType result,
        DType source_premul,
        DType alpha,
        DType acc_alpha)
{
    if (alpha == 1) {
        return (source_premul - result) * acc_alpha;
    }

    DType error = (source_premul - result) / (1 - alpha);

    if (error >= -1 && error <= 0) {
        error = 0;
    }
    if (error < -1) {
        error += 1;
    }

    error *= (1 - alpha);
    error *= acc_alpha;

    return error;
}

template <typename DType>
inline Rgb<DType> error_from_reversed_blend(
        const Rgb<DType>& result,
        const Rgb<DType>& source_premul,
        DType alpha,
        DType acc_alpha)
{
    return {
        channel_error_from_reversed_blend(
                result.r, source_premul.r, alpha, acc_alpha),
        channel_error_from_reversed_blend(
                result.g, source_premul.g, alpha, acc_alpha),
        channel_error_from_reversed_blend(
                result.b, source_premul.b, alpha, acc_alpha),
    };
}

inline constexpr DType max_abs_rgb_error{std::sqrt(3)};

template <typename DType>
inline DType rgb_to_distance(const Rgb<DType>& diff) {
    return std::hypot(diff.r, diff.g, diff.b);
}

template <typename DType>
inline DType rgb_distance(const Rgb<DType>& c1, const Rgb<DType>& c2) {
    const Rgb<DType> diff{c2.r - c1.r, c2.g - c1.g, c2.b - c1.b};

    return rgb_to_distance(diff);
}

// Standard alpha blending formula: R = (1 - A) * D + A * S
// or R = (1 - A) * D + S_premul.
// Here it is transformed thus:
// D = (R - S_premul) / (1 - A).
// R = result, D = destination, S = source, A = alpha.
template <typename DType>
inline DType reverse_blend_channel(DType result, DType source_premul, DType alpha) {
    return alpha == 1
            ? result // Value doesn't matter
            : std::clamp((result - source_premul) / (1 - alpha), 0.0f, 1.0f);
}

template <typename DType>
inline Rgb<DType> reverse_blend(
        const Rgb<DType>& result,
        const Rgb<DType>& source_premul,
        DType alpha)
{
    return {
            reverse_blend_channel(result.r, source_premul.r, alpha),
            reverse_blend_channel(result.g, source_premul.g, alpha),
            reverse_blend_channel(result.b, source_premul.b, alpha),
    };
}

template <typename DType>
inline DType gamma_to_linear(DType c) {
    return c >= 0.04045f ? std::pow((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
}

template <typename DType>
struct Oklab {
    DType l{};
    DType a{};
    DType b{};
};

template <typename DType>
inline Oklab<DType> rgb_to_oklab(const Rgb<DType>& rgb) {
    const DType r_l{gamma_to_linear(rgb.r)};
    const DType g_l{gamma_to_linear(rgb.g)};
    const DType b_l{gamma_to_linear(rgb.b)};

    const DType l{std::cbrt(0.4122214708f * r_l + 0.5363325363f * g_l + 0.0514459929f * b_l)};
    const DType m{std::cbrt(0.2119034982f * r_l + 0.6806995451f * g_l + 0.1073969566f * b_l)};
    const DType s{std::cbrt(0.0883024619f * r_l + 0.2817188376f * g_l + 0.6299787005f * b_l)};

    return {
        0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s,
        1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
        0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s,
    };
}

template <typename DType>
inline DType oklab_distance(const Oklab<DType>& c1, const Oklab<DType>& c2) {
    return std::hypot(c2.a - c1.a, c2.b - c1.b) + std::abs(c2.l - c1.l);
}

template <typename DType>
inline DType hsv_saturation(const Rgb<DType>& rgb) {
    const auto [min, max] = std::minmax({rgb.r, rgb.g, rgb.b});
    return max == 0 ? 0 : 1 - min / max;
}
