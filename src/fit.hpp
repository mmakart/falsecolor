#pragma once

#include "array2d.hpp"
#include "image.hpp"
#include "smudge.hpp"
#include <algorithm>
#include <cmath> // std::abs(long)
#include <numeric>
#include <limits>
#include <utility>
#include <vector>
#include <iostream>

template <typename DType = float> // double is also possible
struct ReversedGreedyFitter {
    ReversedGreedyFitter(
            const Image& target,
            const Image& canvas,
            DType error_tolerance,
            const std::vector<size_t>& allowed_brush_types)
        : m_width(target.width())
        , m_height(target.height())
        , m_threshold_alpha{error_tolerance}
        , m_error_tolerance{error_tolerance}
        , m_allowed_brush_types(allowed_brush_types)
        , m_canvas(canvas)
        , m_target(target)
        , m_oklab_target(m_width, m_height)
        , m_rev_canvas(target)
        , m_rgb_acc_errors(m_width, m_height, {0, 0, 0})
        , m_abs_acc_errors(m_width, m_height, 0)
        , m_abs_errors_from_canvas(m_width, m_height, 0)
        , m_alphas(m_width, m_height, 1)
        , m_forward_table(m_width, m_height)
    {
        std::transform(m_target.cbegin(), m_target.cend(),
                m_oklab_target.begin(), rgb_to_oklab<DType>);
        init_forward_table();
    }

    std::vector<Smudge<DType>> fit() {
        std::vector<Smudge<DType>> result;

        while (true) {
            Smudge<DType> best_smudge{};
            DType best_error{std::numeric_limits<DType>::max()};
            DType best_distance{std::numeric_limits<DType>::max()};
            DType best_alpha_reduced{std::numeric_limits<DType>::min()};
#ifdef CONSIDER_ALPHA
            if (std::none_of(m_alphas.cbegin(), m_alphas.cend(),
                    [this](DType el) {
                        return el > m_threshold_alpha;
                    }))
#else
            if (std::none_of(m_abs_errors_from_canvas.cbegin(),
                    m_abs_errors_from_canvas.cend(),
                    [this](DType el) {
                        return el > m_error_tolerance;
                    }))
#endif
                break;

            // TODO: replace linear search to heap.
            std::for_each(m_forward_table.cbegin(), m_forward_table.cend(),
                [&, x = 0, y = 0](const auto& cell) mutable {
                    for (size_t type_idx : m_allowed_brush_types) {
#ifdef CONSIDER_ALPHA
                        if (cell.threshold_alpha_count_per_brush_type[type_idx] == 0)
#else
                        if (cell.threshold_error_count_per_brush_type[type_idx] == 0)
#endif
                            continue;

                        using PredefinedBrushes::num_colors;
                        for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
                            const DType error{cell
                                    .errors_per_brush_type(color_idx, type_idx)};
                            const DType distance{cell
                                    .distances_per_brush_type(color_idx, type_idx)};
                            const DType alpha_reduced{cell
                                    .alpha_reduced_per_brush_type[type_idx]};

                            // TODO: replace to weighted sum?
                            if ((error < best_error) ||
                                    (error == best_error &&
                                     alpha_reduced > best_alpha_reduced) ||
                                    (error == best_error &&
                                     alpha_reduced == best_alpha_reduced &&
                                     distance < best_distance)) {
                                best_error = error;
                                best_alpha_reduced = alpha_reduced;
                                best_distance = distance;

                                best_smudge = Smudge<DType>{x, y, type_idx, color_idx};
                            }
                        }
                    }
                    if (++x == m_width) {
                        ++y;
                        x = 0;
                    }
                });

            reverse_apply_smudge(best_smudge);

            result.push_back(best_smudge);
        }

        std::reverse(result.begin(), result.end()); // Important

        print_statistics(result);

        return result;
    }

private:
    struct ForwardPixelData {
        // Stored together for memory locality and performance.
        Array2DConstDim<DType,
                PredefinedBrushes::num_colors,
                PredefinedBrushes::num_alphas> excessive_errors_per_alpha{};
        Array2DConstDim<DType,
                PredefinedBrushes::num_colors,
                PredefinedBrushes::num_types> errors_per_brush_type{};
        Array2DConstDim<DType,
                PredefinedBrushes::num_colors,
                PredefinedBrushes::num_types> distances_per_brush_type{};
#ifdef CONSIDER_ALPHA
        std::array<int,
                PredefinedBrushes::num_types> threshold_alpha_count_per_brush_type{};
#else
        std::array<int,
                PredefinedBrushes::num_types> threshold_error_count_per_brush_type{};
#endif
        std::array<DType,
                PredefinedBrushes::num_types> alpha_reduced_per_brush_type{};
    };

    template <typename Func>
    void for_all_image(Func func) {
        for (Signed y = 0; y < m_height; ++y) {
            for (Signed x = 0; x < m_width; ++x) {
                func(x, y);
            }
        }
    }

    template <typename Func>
    void for_smudge_pixels(const Smudge<DType>& smudge, Func func) {
        for (const auto [dx, dy, _, alpha_idx] : smudge.pixels_data()) {
            const auto [x, y] = std::pair{smudge.x + dx, smudge.y + dy};

            if (y < 0 || y >= m_height || x < 0 || x >= m_width) {
                continue;
            }

            func(x, y, smudge.color_idx, alpha_idx);
        }
    }

    template <typename Func>
    void for_smudge_pixels_and_neighborhood(
            const Smudge<DType>& smudge,
            Func func)
    {
        using PredefinedBrushes::all_types;

        const SmudgeProperties<DType>& props{all_types[smudge.type_idx]};

        // Doesn't work correctly with brushes bigger than of plus shape (3x3).
        const Signed rhombic_radius{props.num_pixels == 5 ? 2 : 1};
        const Signed start_dy{-rhombic_radius};
        const Signed end_dy{rhombic_radius + 1};

        for (Signed dy = start_dy; dy < end_dy; ++dy) {
            const Signed start_dx{std::abs(dy) - rhombic_radius};
            const Signed end_dx{1 - start_dx};

            for (Signed dx = start_dx; dx < end_dx; ++dx) {
                const Signed x{smudge.x + dx};
                const Signed y{smudge.y + dy};

                if (y < 0 || y >= m_height || x < 0 || x >= m_width) {
                    continue;
                }

                func(x, y);
            }
        }
    }

    void update_errors_per_alpha(Signed x, Signed y) {
        using PredefinedBrushes::all_alphas, PredefinedBrushes::num_alphas,
                  PredefinedBrushes::num_colors, PredefinedBrushes::premul;

        for (size_t alpha_idx = 0; alpha_idx < num_alphas; ++alpha_idx) {
            const DType alpha{all_alphas[alpha_idx]};

            for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
                const Rgb<DType> rgb_error{error_from_reversed_blend(
                        m_rev_canvas(x, y),
                        premul(alpha_idx, color_idx),
                        alpha,
                        m_alphas(x, y))};
                const Oklab<DType> target_with_error{rgb_to_oklab(m_target(x, y) +
                        rgb_error)};
                const DType abs_error{oklab_distance(m_oklab_target(x, y),
                        target_with_error)};

                m_stats(x, y).excessive_errors_per_alpha(color_idx, alpha_idx) =
                        std::max(static_cast<DType>(0), abs_error +
                        0.5f * // To avoid cascade error spreading to surrounding pixels.
                        m_abs_acc_errors(x, y) - m_error_tolerance);
            }
        }
    }

    void update_distance_per_brush_type(Signed x, Signed y) {
        using PredefinedBrushes::all_types, PredefinedBrushes::all_colors,
                PredefinedBrushes::num_types, PredefinedBrushes::num_colors;

        for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
            const auto& props{all_types[type_idx]};

            for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
                DType sum{0};

                for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {
                    const Signed new_x{x + props.xs[coord_idx]};
                    const Signed new_y{y + props.ys[coord_idx]};

                    if (new_y < 0 || new_y >= m_height ||
                            new_x < 0 || new_x >= m_width) {
                        continue;
                    }

                    // Prefer gray brushes in little saturated target pixels.
                    const DType saturation_coef{1.0f + 7.0f *
                            std::max(static_cast<DType>(0),
                            hsv_saturation(all_colors[color_idx].color) -
                            hsv_saturation(m_target(new_x, new_y)))}; // 1..8

                    // Change to Oklab distance? Probably not because:
                    // 1. Not much difference in result quality.
                    // 2. It'd slow down the fitting algorithm by 2-3 times.
                    sum += rgb_distance(m_rev_canvas(new_x, new_y),
                            all_colors[color_idx].color) * saturation_coef *
                            m_alphas(new_x, new_y);
                }

                m_forward_table(x, y).distances_per_brush_type(color_idx, type_idx) = sum;
            }
        }
    }

    void update_errors_from_canvas(Signed x, Signed y) {
        const Rgb<DType> rgb_error{error_from_reversed_blend(
                m_rev_canvas(x, y),
                m_canvas(x, y),
                static_cast<DType>(1),
                m_alphas(x, y))};

        const Oklab<DType> target_with_error{rgb_to_oklab(m_target(x, y) +
                rgb_error)};

        m_abs_errors_from_canvas(x, y) = oklab_distance(m_oklab_target(x, y),
                target_with_error);
    }

    void update_errors_per_brush_type(Signed x, Signed y) {
        using PredefinedBrushes::all_types, PredefinedBrushes::num_types,
                PredefinedBrushes::num_colors;

        for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
            const auto& props{all_types[type_idx]};

            for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
                DType sum{0};

                for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {
                    const Signed new_x{x + props.xs[coord_idx]};
                    const Signed new_y{y + props.ys[coord_idx]};

                    if (new_y < 0 || new_y >= m_height ||
                            new_x < 0 || new_x >= m_width) {
                        continue;
                    }

                    const size_t alpha_idx{props.alphas_idxs[coord_idx]};

                    sum += m_forward_table(new_x, new_y)
                            .excessive_errors_per_alpha(color_idx, alpha_idx);
                }

                m_forward_table(x, y).errors_per_brush_type(color_idx, type_idx) = sum;
            }
        }
    }

    void update_alpha_reduced_per_brush_type(Signed x, Signed y) {
        using PredefinedBrushes::all_types, PredefinedBrushes::num_types;

        for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
            const auto& props{all_types[type_idx]};

            DType sum{0};
            for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {
                const Signed new_x{x + props.xs[coord_idx]};
                const Signed new_y{y + props.ys[coord_idx]};

                if (new_y < 0 || new_y >= m_height ||
                        new_x < 0 || new_x >= m_width) {
                    continue;
                }

                const DType alpha{props.alphas[coord_idx]};

                sum += m_alphas(new_x, new_y) * alpha;
            }

            m_forward_table(x, y).alpha_reduced_per_brush_type[type_idx] = sum;
        }
    }

#ifdef CONSIDER_ALPHA
    void update_threshold_alpha_count_per_brush_type(Signed x, Signed y) {
        using PredefinedBrushes::all_types, PredefinedBrushes::num_types;

        for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
            const auto& props{all_types[type_idx]};

            int count{0};
            for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {
                const Signed new_x{x + props.xs[coord_idx]};
                const Signed new_y{y + props.ys[coord_idx]};

                if (new_y < 0 || new_y >= m_height ||
                        new_x < 0 || new_x >= m_width) {
                    continue;
                }

                if (m_alphas(new_x, new_y) > m_threshold_alpha) {
                    ++count;
                }
            }

            m_forward_table(x, y).threshold_alpha_count_per_brush_type[type_idx] = count;
        }
    }
#else
    void update_threshold_error_count_per_brush_type(Signed x, Signed y) {
        using PredefinedBrushes::all_types, PredefinedBrushes::num_types;

        for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
            const auto& props{all_types[type_idx]};

            int count{0};
            for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {
                const Signed new_x{x + props.xs[coord_idx]};
                const Signed new_y{y + props.ys[coord_idx]};

                if (new_y < 0 || new_y >= m_height ||
                        new_x < 0 || new_x >= m_width) {
                    continue;
                }

                if (m_abs_errors_from_canvas(new_x, new_y) > m_error_tolerance) {
                    ++count;
                }
            }

            m_forward_table(x, y).threshold_error_count_per_brush_type[type_idx] = count;
        }
    }
#endif

    void update_acc_errors(Signed x, Signed y, size_t color_idx, size_t alpha_idx) {
        using PredefinedBrushes::premul, PredefinedBrushes::all_alphas;

        const Rgb<DType> rgb_error{error_from_reversed_blend(
                m_rev_canvas(x, y),
                premul(alpha_idx, color_idx),
                all_alphas[alpha_idx],
                m_alphas(x, y))};

        m_rgb_acc_errors(x, y) += rgb_error;

        const Oklab<DType> target_with_error{rgb_to_oklab(m_target(x, y) +
                m_rgb_acc_errors(x, y))};
        const DType abs_error{oklab_distance(m_oklab_target(x, y),
                target_with_error)};
        m_abs_acc_errors(x, y) = abs_error;
    }

    void update_rev_canvas_and_alpha(
            Signed x,
            Signed y,
            size_t color_idx,
            size_t alpha_idx)
    {
        using PredefinedBrushes::premul, PredefinedBrushes::all_alphas;

        const DType alpha{all_alphas[alpha_idx]};
        const Rgb<DType> color_premul{premul(alpha_idx, color_idx)};

        m_rev_canvas(x, y) = reverse_blend(m_rev_canvas(x, y), color_premul, alpha);
        m_alphas(x, y) *= 1 - alpha;
    }

    void init_forward_table() {
        for_all_image([this](Signed x, Signed y) {
            update_errors_per_alpha(x, y);
        });

        for_all_image([this](Signed x, Signed y) {
            update_errors_from_canvas(x, y);
        });

        // Must follow "for all image: update errors per alpha"
        for_all_image([this](Signed x, Signed y) {
            update_errors_per_brush_type(x, y);
        });

        for_all_image([this](Signed x, Signed y) {
            update_distance_per_brush_type(x, y);
        });

#ifdef CONSIDER_ALPHA
        for_all_image([this](Signed x, Signed y) {
            update_threshold_alpha_count_per_brush_type(x, y);
        });
#else
        // Must follow "for all image: update errors from canvas".
        for_all_image([this](Signed x, Signed y) {
            update_threshold_error_count_per_brush_type(x, y);
        });
#endif

        for_all_image([this](Signed x, Signed y) {
            update_alpha_reduced_per_brush_type(x, y);
        });
    }

    void reverse_apply_smudge(const Smudge<DType>& smudge) {
        // Must preceide updating of m_alphas because it needs its old values.
        for_smudge_pixels(smudge, [this](Signed x, Signed y,
                        size_t color_idx, size_t alpha_idx) {
                update_acc_errors(x, y, color_idx, alpha_idx);
            });

        for_smudge_pixels(smudge, [this](Signed x, Signed y,
                        size_t color_idx, size_t alpha_idx) {
                update_rev_canvas_and_alphas(x, y, color_idx, alpha_idx);
            });

        // update_acc_errors and update_rev_canvas_and_alphas must
        // preceide everything below:

        for_smudge_pixels(smudge, [this](Signed x, Signed y, size_t, size_t) {
                update_errors_per_alpha(x, y);
            });

        for_smudge_pixels(smudge, [this](Signed x, Signed y, const Rgb<DType>&, DType) {
                update_errors_from_canvas(x, y);
            });

        for_smudge_pixels_and_neighborhood(smudge, [this](Signed x, Signed y) {
                update_distance_per_brush_type(x, y);
            });

        // Must follow "for smudge pixels: update errors per alpha"
        for_smudge_pixels_and_neighborhood(smudge, [this](Signed x, Signed y) {
                update_errors_per_brush_type(x, y);
            });

#ifdef CONSIDER_ALPHA
        for_smudge_pixels_and_neighborhood(smudge, [this](Signed x, Signed y) {
                update_threshold_alpha_count_per_brush_type(x, y);
            });
#else
        // Must follow "for smudge pixels: update errors if canvas"
        for_smudge_pixels_and_neighborhood(smudge, [this](Signed x, Signed y) {
                update_threshold_error_count_per_brush_type(x, y);
            });
#endif
        for_smudge_pixels_and_neighborhood(smudge, [this](Signed x, Signed y) {
                update_alpha_reduced_per_brush_type(x, y);
            });
    }

    void print_statistics(const std::vector<Smudge<DType>>& result) {
        const double introduced{std::accumulate(m_abs_acc_errors.cbegin(),
                m_abs_acc_errors.cend(), 0.0)};

        const double from_canvas{std::accumulate(m_abs_errors_from_canvas.cbegin(),
                m_abs_errors_from_canvas.cend(), 0.0)};

        std::cerr << "Error introduced = " << introduced << '\n';
        std::cerr << "Error from canvas = " << from_canvas << '\n';
        std::cerr << "Total error = " << introduced + from_canvas << '\n';

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

    const Signed m_width;
    const Signed m_height;

    const DType m_threshold_alpha{};
    const DType m_error_tolerance{};

    const std::vector<size_t> m_allowed_brush_types;

    // Reference image of in-game canvas (e.g. blank or partially finished one).
    const Array2D<Rgb<DType>> m_canvas;

    // Reference image of the target image.
    const Array2D<Rgb<DType>> m_target;

    Array2D<Oklab<DType>> m_oklab_target; // Should be const.

    // Working area for reversed applying of smudges.
    // Initialy is the same as the target image.
    Array2D<Rgb<DType>> m_rev_canvas;

    // Errors per pixel introduced during reversed applyments of smudges.
    // At the moment of the algorithm's end it shows the pixel-by-pixel RGB
    // part of the difference between result image and the target introduced
    // by smudges.
    // Initially is {0,0,0} everywhere. Can reach down to {-1,-1,-1} or up to
    // {1,1,1} in extreme cases. The closer to {0,0,0}, the better.
    // The key feature is that by design of the algorithm the elements of this
    // matrix cannot ever get closer to the origin, but can only move away or
    // stay unchanged. This means it's impossible to fix any error introduced
    // by particular reversed smudge.
    Array2D<Rgb<DType>> m_rgb_acc_errors;

    // Grayscale variation of m_rgb_acc_errors (e.g. euclidean RGB distance).
    // Currently it's OkLAB distance between target and target with error.
    Array2D<DType> m_abs_acc_errors;

    // If reverse-apply or "steal" the pixel from in-game canvas.
    // It allows to avoid applying unnecessary smudges where canvas pixels
    // have close enough color to the corresponding target image ones.
    // Also useful for calculating of correction smudge sequence if a mistake
    // was made in the game.
    Array2D<DType> m_abs_errors_from_canvas;

    // Reflects how much next reversed smudge can affect resulting color
    // or add error to m_rgb_acc_errors. Geometrically drops to 0 on each
    // reversed applyment of a smudge (the more alpha the smudge pixel has
    // the more it drops). Initially is 1 everywhere.
    Array2D<DType> m_alphas;

    // Necessary data for picking next smudge.
    Array2D<ForwardPixelData> m_forward_table;
};

// Gives a list of smudges (x, y, type, color) which, applied in this order,
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
