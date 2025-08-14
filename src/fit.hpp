#pragma once

#include "array2d.hpp"
#include "image.hpp"
#include "smudge.hpp"
#include <algorithm>
#include <cmath> // std::abs(long)
#include <numeric>
#include <vector>
#include <iostream> //TODO temp

template <typename DType = float> // double is also possible
struct ReversedGreedyFitter {
    ReversedGreedyFitter(
            const Image& target,
            const Image& canvas,
            DType error_tolerance,
            const std::vector<size_t>& allowed_brush_types
            )
        : m_canvas(canvas)
        , m_target(target)
        , m_rev_canvas(target)
        , m_errors_added(target.width(), target.height(), {0, 0, 0})
        , m_abs_errors_added(target.width(), target.height(), 0)
        , m_abs_errors_if_canvas(target.width(), target.height(), 0)
        , m_acc_alphas(target.width(), target.height(), 1)
        , m_stats(target.width(), target.height())
        , m_threshold_alpha{error_tolerance}
        , m_allowed_brush_types(allowed_brush_types)
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
        //std::vector<size_t> type_idxs(3); //TODO temp

        std::generate(coords.begin(), coords.end(), [&, xx=0, yy=0] () mutable {
                    if (xx == m_stats.width()) { xx = 0; ++yy; }
                    Coord res{xx, yy};
                    ++xx;
                    return res;
                }
        );
        std::iota(color_idxs.begin(), color_idxs.end(), 0);
        //std::iota(type_idxs.begin(), type_idxs.end(), 0); //TODO temp

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
                //for (size_t type_idx : type_idxs) { // TODO: temp
                for (size_t type_idx : m_allowed_brush_types) {
                    const bool enough_acc_alpha {
                        m_stats(x, y).threshold_count_per_brush_type[type_idx] > 0
                    };

                    if (!enough_acc_alpha) {
                        continue;
                    }

                    for (size_t color_idx : color_idxs) {
                        const DType error = m_stats(x, y).errors_per_brush_type(type_idx, color_idx);
                        const DType distance = m_stats(x, y).distances_per_brush_type(type_idx, color_idx);

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

    struct PixelStats {
        // Stored on stack for memory locality and performance.
        Array2DConstDim<DType,
                PredefinedBrushes::num_alphas,
                PredefinedBrushes::num_colors> errors_per_alpha{};
        Array2DConstDim<DType,
                PredefinedBrushes::num_types,
                PredefinedBrushes::num_colors> errors_per_brush_type{};
        Array2DConstDim<DType,
                PredefinedBrushes::num_types,
                PredefinedBrushes::num_colors> distances_per_brush_type{};
        std::array<int,
                PredefinedBrushes::num_types> threshold_count_per_brush_type{};
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
                        const Rgb<DType> rgb_error = error_from_reversed_blend(
                                m_rev_canvas(x, y),
                                premul(alpha_idx, color_idx),
                                all_alphas[alpha_idx],
                                m_acc_alphas(x, y)
                        );

                        m_stats(x, y).errors_per_alpha(alpha_idx, color_idx) = rgb_to_distance(rgb_error);
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

                        m_stats(x, y).distances_per_brush_type(type_idx, color_idx) = sum;
                    } // end distances per brush type

                } // end per color

                m_abs_errors_if_canvas(x, y) = rgb_to_distance(
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

                            sum += m_stats(new_x, new_y).errors_per_alpha(alpha_idx, color_idx);
                        }

                        m_stats(x, y).errors_per_brush_type(type_idx, color_idx) = sum;
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
            const auto color{premul(alpha_idx, color_idx)};

            const Rgb<DType> error = error_from_reversed_blend(
                    m_rev_canvas(new_x, new_y),
                    color,
                    alpha,
                    m_acc_alphas(new_x, new_y)
            );
            m_errors_added(new_x, new_y) += error;
            m_abs_errors_added(new_x, new_y) += rgb_to_distance(error);
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

                    const Rgb<DType> rgb_error = error_from_reversed_blend(
                            m_rev_canvas(new_x, new_y),
                            premul(alpha_idx, color_idx),
                            all_alphas[alpha_idx],
                            m_acc_alphas(new_x, new_y)
                    );

                    m_stats(new_x, new_y).errors_per_alpha(alpha_idx, color_idx) =
                            rgb_to_distance(rgb_error);
                }
            }

            m_abs_errors_if_canvas(new_x, new_y) = rgb_to_distance(
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

                        m_stats(rhombic_x, rhombic_y).distances_per_brush_type(type_idx, color_idx) = sum;
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
                            sum += m_stats(new_x, new_y).errors_per_alpha(alpha_idx, color_idx);
                        }

                        m_stats(rhombic_x, rhombic_y).errors_per_brush_type(type_idx, color_idx) = sum;
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
        std::cerr << "Total error = " << std::accumulate(m_abs_errors_added.cbegin(), //TODO: add m_abs_errors_if_canvas here
                m_abs_errors_added.cend(), 0.0) << '\n';
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

    // Reference image of in-game canvas (e.g. blank or partially finished one)
    const Array2D<Rgb<DType>> m_canvas;

    // Reference image of the target image
    const Array2D<Rgb<DType>> m_target;

    // Working area for reversed applying of smudges.
    // Initialy is the same as the target image.
    Array2D<Rgb<DType>> m_rev_canvas;

    // Errors per pixel introduced during reversed applyments of smudges.
    // Initially is 0 everywhere. Can reach down to -1 or up to 1 during
    // calculations in extreme cases. The closer to 0, the better.
    Array2D<Rgb<DType>> m_errors_added;

    // Grayscale variation of m_errors_added (e.g. euclidean RGB distance)
    Array2D<DType> m_abs_errors_added;

    // If reverse-apply or "steal" the pixel from in-game canvas.
    // It allows to avoid applying unnecessary smudges where canvas pixels
    // have close enough color to the corresponding target image ones.
    // Also useful for calculating of correction smudge sequence if a mistake
    // was made in the game.
    Array2D<DType> m_abs_errors_if_canvas;

    // Reflects how much next reversed smudge can affect resulting color
    // or add error to m_errors_added. Geometrically drops to 0 on each
    // reversed applyment of a smudge (the more alpha the smudge pixel has
    // the more it drops). Initially is 1 everywhere.
    Array2D<DType> m_acc_alphas;

    // Necessary data for picking next smudge
    Array2D<PixelStats> m_stats;

    const DType m_threshold_alpha{};

    const std::vector<size_t> m_allowed_brush_types;
};

// Gives a list of smudges (x, y, color, type) which, applied in this order,
// transform the current state ('canvas') to as close as possible
// to desired one ('target').
template <typename DType>
std::vector<Smudge<DType>> fit_target_image(
        const Image& target,
        const Image& canvas,
        const DType error_tolerance,
        const std::vector<size_t>& types_allowed)
{
    ReversedGreedyFitter<DType> fitter(target, canvas, error_tolerance, types_allowed);

    return fitter.fit();
}
