#pragma once

#include "image.hpp"
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
    using Signed = ptrdiff_t;

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
    using DType = float;
//    using DType = Settings::DType; // TODO add a single source of information?

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

    inline constexpr size_t num_premul{num_colors * num_alphas};

    inline constexpr std::array<Rgb<DType>, num_premul> init_premul() {
        std::array<Rgb<DType>, num_premul> result{};

        for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
            for (size_t alpha_idx = 0; alpha_idx < num_alphas; ++alpha_idx) {
                auto color = all_colors[color_idx].color;
                auto alpha = all_alphas[alpha_idx];

                const size_t index{color_idx * num_alphas + alpha_idx};
                result[index] = {
                        color.r * alpha, color.g * alpha, color.b * alpha
                };
            }
        }

        return result;
    }

    inline constexpr std::array<Rgb<DType>, num_premul> premul{init_premul()};

    // Same order as in enum BrushType!
    inline constexpr std::array all_types{pixel_props, water_props, oil_props};
}

template <typename DType>
struct Smudge {
    using Signed = ptrdiff_t;

    Signed x{};
    Signed y{};
    size_t type_idx{};
    size_t color_idx{};

    void apply(Image& image) const
    {
        if (x < 0 || static_cast<size_t>(x) >= image.width() ||
                y < 0 || static_cast<size_t>(y) >= image.height()) {
            return;
        }

        const auto& props{PredefinedBrushes::all_types[type_idx]};
        const auto color{PredefinedBrushes::all_colors[color_idx].color};

        for (auto i = 0; i < props.num_pixels; ++i) {
            image.blend_pixel(
                    x + props.xs[i],
                    y + props.ys[i],
                    color,
                    props.alphas[i]
            );
        }
    }
};

//TODO remove?
template <typename DType>
struct RankedSmudge {
    Smudge<DType> smudge{};

    DType distance{};
    DType error{};
};

// Standard alpha blending formula: R = (1 - A) * D + A * S
// or R = (1 - A) * D + S_premul.
// Possible return value is in range [-1; 1].
// result and source_premul are expected to be in range [-1; 1].
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

template <typename DType>
inline DType rgb_to_distance(const Rgb<DType>& diff) {
    return std::hypot(diff.r, diff.g, diff.b);
}

template <typename DType>
inline DType color_dist(const Rgb<DType>& c1, const Rgb<DType>& c2) {
    const Rgb<DType> diff{c2.r - c1.r, c2.g - c1.g, c2.b - c1.b};

    return rgb_to_distance(diff);
}

// Standard alpha blending formula: R = (1 - A) * D + A * S
// or R = (1 - A) * D + S_premul.
// Here it is ised thus:
// D = (R - S_premul) / (1 - A).
// R is result, D is destination, S is source, A is alpha.
template <typename DType>
inline DType reverse_blend_channel(DType result, DType source_premul, DType alpha) {
    return alpha == 1
            ? result // value doesn't matter
            : std::clamp((result - source_premul) / (1 - alpha),
                    static_cast<DType>(0),
                    static_cast<DType>(1));
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
