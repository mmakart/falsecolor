#pragma once

#include "smudge.hpp"
#include "image.hpp"
#include <algorithm>
#include <cmath> // std::abs(long)
#include <numeric>
#include <vector>
#include <iostream> //TODO temp

template <typename DType = float> // double is also possible
struct ReversedGreedyFitter {
    using Signed = ptrdiff_t; // To avoid unsigned-signed indexing bugs

    ReversedGreedyFitter(const Image& target,
            const Image& canvas, DType error_tolerance)
        : m_canvas(canvas)
        , m_target(target)
        , m_rev_canvas(target)
        , m_errors(target.width(), target.height(), {0, 0, 0})
        , m_abs_errors(target.width(), target.height(), 0)
        , m_acc_alphas(target.width(), target.height(), 1)
        , m_stats(target.width(), target.height())
        , m_threshold_alpha{error_tolerance}
    {
        init_pixel_stats();
    }

    std::vector<Smudge<DType>> fit() {
        std::vector<Smudge<DType>> result;

        int sum_threshold_alpha{};

        //TODO temp
        struct Coord { Signed x, y; };
        std::vector<Coord> coords(m_stats.width() * m_stats.height());
        std::vector<size_t> color_idxs(PredefinedBrushes::num_colors);
        std::vector<size_t> type_idxs(3); //TODO temp
        //std::vector<size_t> type_idxs{1};

        std::generate(coords.begin(), coords.end(), [&, xx=0, yy=0] () mutable {
                    if (xx == m_stats.width()) { xx = 0; ++yy; }
                    Coord res{xx, yy};
                    ++xx;
                    return res;
                }
        );
        std::iota(color_idxs.begin(), color_idxs.end(), 0);
        std::iota(type_idxs.begin(), type_idxs.end(), 0); //TODO temp

        while (true) {
            Signed best_x{0}, best_y{0};
            size_t best_type_idx{}, best_color_idx{};
            DType best_error{1000000}, best_distance{1000000};

            sum_threshold_alpha = std::count_if(m_acc_alphas.cbegin(), m_acc_alphas.cend(),
                    [&](const auto& el) {
                        return el > m_threshold_alpha;
                    }
            );

            if (sum_threshold_alpha == 0) {
                break;
            }

            for (const auto& coord : coords) {
                Signed x = coord.x;
                Signed y = coord.y;
                for (size_t type_idx : type_idxs) {
                    const bool enough_acc_alpha {
                        m_stats(x, y).threshold_count_per_brush_type[type_idx] > 0
                    };

                    if (!enough_acc_alpha) {
                        continue;
                    }

                    for (size_t color_idx : color_idxs) {
                        using PredefinedBrushes::num_colors, PredefinedBrushes::num_types;

                        const size_t index{num_types * color_idx + type_idx};

                        const DType error = m_stats(x, y).errors_per_brush_type[index];
                        const DType distance = m_stats(x, y).distances_per_brush_type[index];

                        if (error < best_error
                                || error == best_error && distance < best_distance) {
                            best_error = error;
                            best_distance = distance;

                            best_x = x;
                            best_y = y;
                            best_type_idx = type_idx;
                            best_color_idx = color_idx;
                        }
                    }
                }
            }

            Smudge<DType> best_smudge{best_x, best_y, best_type_idx, best_color_idx};

            reverse_apply_smudge(best_smudge);

            result.push_back(best_smudge);
        }

        std::reverse(result.begin(), result.end()); // Important

        print_statistics(result);

        return result;
    }

    template <typename T>
    class Array2D {
    public:
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
            static_assert(std::is_same_v<T, Rgb<DType>> == true);

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

    struct PixelStats {
        static constexpr size_t s_alphas_size {
                PredefinedBrushes::num_colors * PredefinedBrushes::num_alphas
        };

        static constexpr size_t s_types_size {
                PredefinedBrushes::num_colors * PredefinedBrushes::num_types
        };

        static constexpr size_t s_distances_size {
                PredefinedBrushes::num_colors * PredefinedBrushes::num_types
        };

        static constexpr size_t s_threshold_count_size {
                PredefinedBrushes::num_types
        };

        // Not std::vector because it would acquire random memory
        // from the heap which would hurt memory locality and drop performance.
        std::array<DType, s_alphas_size> errors_per_alpha{};
        std::array<DType, s_types_size> errors_per_brush_type{};
        std::array<DType, s_distances_size> distances_per_brush_type{};
        std::array<int, s_threshold_count_size> threshold_count_per_brush_type{};

        // If "steal" pixel from in-game canvas.
        // It allows to avoid applying unnecessary smudges where canvas pixels
        // have close enough color to the corresponding target image ones.
        // Also useful if fixing player mistakes which can occur while actual
        // drawing to have a chance to not redraw all canvas from ground up
        // again.
        DType error_if_take_from_canvas{};
    };

private:
    void init_pixel_stats() {
        using PredefinedBrushes::num_colors, PredefinedBrushes::num_alphas,
                PredefinedBrushes::num_types, PredefinedBrushes::premul,
                PredefinedBrushes::all_types, PredefinedBrushes::all_alphas,
                PredefinedBrushes::all_colors;

        // 1. Errors per alpha.
        // 2. Distances per brush type.
        for (Signed y = 0; y < m_stats.height(); ++y) {
            for (Signed x = 0; x < m_stats.width(); ++x) {
                // Per color
                for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
                    // Errors per alpha
                    for (size_t alpha_idx = 0; alpha_idx < num_alphas; ++alpha_idx) {
                        const size_t index{color_idx * num_alphas + alpha_idx};

                        const Rgb<DType> rgb_error = error_from_reversed_blend(
                                m_rev_canvas(x, y),
                                premul[index],
                                all_alphas[alpha_idx],
                                m_acc_alphas(x, y)
                        );

                        m_stats(x, y).errors_per_alpha[index] = rgb_to_distance(rgb_error);
                    } // end errors per alpha

                    // Distances per brush type
                    for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
                        // properties
                        const auto& props{all_types[type_idx]};

                        DType sum{0};
                        for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {
                            const auto new_x{x + props.xs[coord_idx]};
                            const auto new_y{y + props.ys[coord_idx]};

                            if (new_x < 0 || new_x >= m_stats.width()
                                    || new_y < 0 || new_y >= m_stats.height()) {
                                continue;
                            }

                            sum += color_dist(
                                    m_rev_canvas(new_x, new_y),
                                    all_colors[color_idx].color
                            ) * m_acc_alphas(new_x, new_y);
                        }

                        const size_t index{color_idx * num_types + type_idx};

                        m_stats(x, y).distances_per_brush_type[index] = sum;
                    } // end distances per brush type

                } // end per color

                m_stats(x, y).error_if_take_from_canvas = rgb_to_distance(
                        error_from_reversed_blend(
                            m_rev_canvas(x, y),
                            m_canvas(x, y),
                            static_cast<DType>(1),
                            m_acc_alphas(x, y)
                ));
            }
        }

        // 3. Count of pixels per brush type where accumulated alpha 
        // is greater than threshold alpha
        for (Signed y = 0; y < m_stats.height(); ++y) {
            for (Signed x = 0; x < m_stats.width(); ++x) {
                for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
                    // properties
                    const auto& props{all_types[type_idx]};

                    int count{0};
                    for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {
                        const auto new_x{x + props.xs[coord_idx]};
                        const auto new_y{y + props.ys[coord_idx]};

                        if (new_x < 0 || new_x >= m_stats.width()
                                || new_y < 0 || new_y >= m_stats.height()) {
                            continue;
                        }

                        if (m_acc_alphas(new_x, new_y) > m_threshold_alpha) {
                            ++count;
                        }
                    }

                    m_stats(x, y).threshold_count_per_brush_type[type_idx] = count;
                }
            }
        }

        // 4. Errors per brush type
        // Must be calculated only after "1. Errors per alpha"
        // made for each (x, y).
        for (Signed y = 0; y < m_stats.height(); ++y) {
            for (Signed x = 0; x < m_stats.width(); ++x) {
                // Per color
                for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
                    // Errors per brush type
                    for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
                        // properties
                        const auto& props{all_types[type_idx]};

                        DType sum{0};
                        for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {

                            const auto new_x{x + props.xs[coord_idx]};
                            const auto new_y{y + props.ys[coord_idx]};

                            if (new_x < 0 || new_x >= m_stats.width()
                                    || new_y < 0 || new_y >= m_stats.height()) {
                                continue;
                            }

                            const size_t alpha_idx{props.alphas_idxs[coord_idx]};
                            const size_t index{num_alphas * color_idx + alpha_idx};

                            sum += m_stats(new_x, new_y).errors_per_alpha[index];
                        }

                        const size_t index{color_idx * num_types + type_idx};

                        m_stats(x, y).errors_per_brush_type[index] = sum;
                    } // end errors per brush_type
                } // end per color
            }
        }
    }

    void reverse_apply_smudge(const Smudge<DType>& smudge) {
        using PredefinedBrushes::all_types, PredefinedBrushes::premul,
                PredefinedBrushes::all_alphas, PredefinedBrushes::num_alphas,
                PredefinedBrushes::num_types, PredefinedBrushes::all_colors,
                PredefinedBrushes::num_colors;

        const auto& type{all_types[smudge.type_idx]};
        const auto x{smudge.x};
        const auto y{smudge.y};
        const size_t color_idx{smudge.color_idx};

        // 1. Updating rev_canvas, acc_alphas, errors, abs_errors
        for (size_t coord_idx = 0; coord_idx < type.num_pixels; ++coord_idx) {
            const auto new_x{x + type.xs[coord_idx]};
            const auto new_y{y + type.ys[coord_idx]};

            if (new_x < 0 || new_x >= m_rev_canvas.width()
                    || new_y < 0 || new_y >= m_rev_canvas.height()) {
                continue;
            }

            const size_t alpha_idx{type.alphas_idxs[coord_idx]};
            const DType alpha{type.alphas[coord_idx]};
            const auto color{premul[num_alphas * color_idx + alpha_idx]};

            const Rgb<DType> error = error_from_reversed_blend(
                    m_rev_canvas(new_x, new_y),
                    color,
                    alpha,
                    m_acc_alphas(new_x, new_y)
            );
            m_errors(new_x, new_y) += error;
            m_abs_errors(new_x, new_y) += rgb_to_distance(error);
            m_rev_canvas(new_x, new_y) = reverse_blend(
                    m_rev_canvas(new_x, new_y), color, alpha);
            m_acc_alphas(new_x, new_y) *= 1 - alpha;
        }

        // 2. Updating alpha errors in m_stats in rhombic area 1x1 or 3x3
        // depending on size of just applied brush.
        for (size_t coord_idx = 0; coord_idx < type.num_pixels; ++coord_idx) {
            const auto new_x{x + type.xs[coord_idx]};
            const auto new_y{y + type.ys[coord_idx]};

            if (new_x < 0 || new_x >= m_stats.width()
                    || new_y < 0 || new_y >= m_stats.height()) {
                continue;
            }

            // Errors per alpha
            for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
                for (size_t alpha_idx = 0; alpha_idx < num_alphas; ++alpha_idx) {

                    const size_t index{color_idx * num_alphas + alpha_idx};

                    const Rgb<DType> rgb_error = error_from_reversed_blend(
                            m_rev_canvas(new_x, new_y),
                            premul[index],
                            all_alphas[alpha_idx],
                            m_acc_alphas(new_x, new_y)
                    );

                    m_stats(new_x, new_y).errors_per_alpha[index] =
                            rgb_to_distance(rgb_error);
                }
            }

            m_stats(new_x, new_y).error_if_take_from_canvas = rgb_to_distance(
                    error_from_reversed_blend(
                        m_rev_canvas(new_x, new_y),
                        m_canvas(new_x, new_y),
                        static_cast<DType>(1),
                        m_acc_alphas(new_x, new_y)
            ));
        }

        // 3. Updating distances and errors per brush type in m_stats
        // (they happened to have equal indexing ranges)
        // in Von Neimann (rhombic) area 3x3 or 5x5
        // depending on just applied smudge size (1x1 or 3x3 resp.).
        const Signed rhombic_radius{type.num_pixels == 5 ? 2 : 1};
        const Signed start_dy{-rhombic_radius};
        const Signed end_dy{rhombic_radius + 1};

        for (Signed dy = start_dy; dy < end_dy; ++dy) {
            const Signed start_dx{std::abs(dy) - rhombic_radius};
            const Signed end_dx{1 - start_dx};

            for (Signed dx = start_dx; dx < end_dx; ++dx) {
                const auto rhombic_x{x + dx};
                const auto rhombic_y{y + dy};

                if (rhombic_x < 0 || rhombic_x >= m_stats.width()
                        || rhombic_y < 0 || rhombic_y >= m_stats.height()) {
                    continue;
                }

                for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {

                    // Distances per brush type
                    for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
                        // properties
                        const auto& props{all_types[type_idx]};

                        DType sum{0};
                        for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {
                            const auto new_x{rhombic_x + props.xs[coord_idx]};
                            const auto new_y{rhombic_y + props.ys[coord_idx]};

                            if (new_x < 0 || new_x >= m_stats.width()
                                    || new_y < 0 || new_y >= m_stats.height()) {
                                continue;
                            }

                            sum += color_dist(
                                    m_rev_canvas(new_x, new_y),
                                    all_colors[color_idx].color
                            ) * m_acc_alphas(new_x, new_y);
                        }

                        const size_t index{color_idx * num_types + type_idx};

                        m_stats(rhombic_x, rhombic_y).distances_per_brush_type[index] = sum;
                    }

                    // Errors per brush type
                    for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
                        // properties
                        const auto& props{all_types[type_idx]};

                        DType sum{0};
                        for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {
                            const auto new_x{rhombic_x + props.xs[coord_idx]};
                            const auto new_y{rhombic_y + props.ys[coord_idx]};

                            if (new_x < 0 || new_x >= m_stats.width()
                                    || new_y < 0 || new_y >= m_stats.height()) {
                                continue;
                            }

                            const size_t alpha_idx{props.alphas_idxs[coord_idx]};
                            const size_t index{num_alphas * color_idx + alpha_idx};

                            sum += m_stats(new_x, new_y).errors_per_alpha[index];
                        }

                        const size_t index{color_idx * num_types + type_idx};

                        m_stats(rhombic_x, rhombic_y).errors_per_brush_type[index] = sum;
                    }
                }

                // Count of pixels per brush type where accumulated alpha
                // is greater than threshold.
                for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
                    // properties
                    const auto& props{all_types[type_idx]};

                    int count{0};
                    for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {
                        const auto new_x{rhombic_x + props.xs[coord_idx]};
                        const auto new_y{rhombic_y + props.ys[coord_idx]};

                        if (new_x < 0 || new_x >= m_stats.width()
                                || new_y < 0 || new_y >= m_stats.height()) {
                            continue;
                        }

                        if (m_acc_alphas(new_x, new_y) > m_threshold_alpha) {
                            ++count;
                        }
                    }

                    m_stats(rhombic_x, rhombic_y).threshold_count_per_brush_type[type_idx] = count;
                }
            }
        }
    }

    void print_statistics(const std::vector<Smudge<DType>>& result) {
        std::cerr << "Total error = " << std::accumulate(m_abs_errors.cbegin(),
                m_abs_errors.cend(), 0.0) << '\n';
        std::cerr << "1 pixel brush count = "
                << std::count_if(result.cbegin(), result.cend(),
                [](const auto& smudge) {
                    return smudge.type_idx == 0;
                }) << '\n';
        std::cerr << "water brush count = " << std::count_if(result.cbegin(), result.cend(),
                [](const auto& smudge) {
                    return smudge.type_idx == 1;
                }) << '\n';
        std::cerr << "oil brush count = " << std::count_if(result.cbegin(), result.cend(),
                [](const auto& smudge) {
                    return smudge.type_idx == 2;
                }) << '\n';
        std::cerr << "Total smudges = " << std::size(result) << '\n';
    }

    // Image RGB data of in-game canvas (e.g. blank or partially finished one)
    const Array2D<Rgb<DType>> m_canvas;

    const Array2D<Rgb<DType>> m_target;

    Array2D<Rgb<DType>> m_rev_canvas;

    // Inavoidable errors per pixel (per channel). Initially is 0 everywhere.
    // Can reach down to -1 or up to 1 during calculations in extreme cases.
    // The closer to 0, the better.
    Array2D<Rgb<DType>> m_errors;

    // Grayscale variation of m_errors (e.g. euclidean RGB distance)
    Array2D<DType> m_abs_errors;

    // Initially is 1 everywhere. Geometrically drops to 0 on each reversed
    // applyment of a smudge (the more alpha a smudge has the more it drops).
    // Reflects how much next reversed smudge can affect resulting color
    // or add error.
    Array2D<DType> m_acc_alphas;

    // Precalculated data to pick a next smudge by custom rank
    Array2D<PixelStats> m_stats;

    const DType m_threshold_alpha{};
};

// Gives a list of smudges (x, y, color, type) which, applied in this order,
// transform the current state ('canvas') to as close as possible
// to desired one ('target').
template <typename DType>
std::vector<Smudge<DType>> fit_target_image(
        const Image& target,
        const Image& canvas,
        const DType error_tolerance,
        const std::vector<SmudgeProperties<DType>>& types_allowed) //TODO add use
{
    ReversedGreedyFitter<DType> fitter(target, canvas, error_tolerance);

    return fitter.fit();
}
