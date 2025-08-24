#pragma once

#include "array2d.hpp"
#include "image.hpp"
#include "smudge.hpp"
#include <algorithm>
#include <cmath> // std::abs(long)
#include <numeric>
#include <utility>
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
        : m_width(target.width())
        , m_height(target.height())
        , m_threshold_alpha{error_tolerance}
        , m_allowed_brush_types(allowed_brush_types)
        , m_canvas(canvas)
        , m_target(target)
        , m_rev_canvas(target)
        , m_errors_added(target.width(), target.height(), {0, 0, 0})
        , m_abs_errors_added(target.width(), target.height(), 0)
        , m_abs_errors_if_canvas(target.width(), target.height(), 0)
        , m_acc_alphas(target.width(), target.height(), 1)
        , m_stats(target.width(), target.height())
    {
        init_pixel_stats();
    }

    std::vector<Smudge<DType>> fit() {
        std::vector<Smudge<DType>> result;

        int num_pixels_to_process{};

        while (true) {
            Signed best_x{0}, best_y{0};
            size_t best_type_idx{}, best_color_idx{};
            DType best_error{1000000}, best_distance{1000000},
                    best_acc_alpha_reduced{-1000000};

            num_pixels_to_process = std::count_if(
                    m_abs_errors_if_canvas.cbegin(),
                    m_abs_errors_if_canvas.cend(),
                    [this](DType el) {
                        // m_threshold_alpha means "max OK error per pixel"
                        return el > m_threshold_alpha;
                    }
            );

            if (num_pixels_to_process == 0) {
                break;
            }

            for (Signed y = 0; y < m_height; ++y) {
                for (Signed x = 0; x < m_width; ++x) {
                    for (size_t type_idx : m_allowed_brush_types) {
                        const bool small_error_here {
                            m_stats(x, y).above_threshold_error_count_per_brush_type[type_idx] == 0
                        };

                        if (small_error_here) {
                            continue;
                        }

                        using PredefinedBrushes::num_colors;
                        for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
                            const DType error {
                                    m_stats(x, y).errors_per_brush_type(type_idx, color_idx)
                            };
                            const DType distance {
                                    m_stats(x, y).distances_per_brush_type(type_idx, color_idx)
                            };
                            const DType acc_alpha_reduced {
                                    m_stats(x, y).acc_alpha_reduced_per_brush_type[type_idx]
                            };

                            // TODO: replace to weighted sum?
                            if ((error < best_error) ||
                                    (error == best_error &&
                                    distance < best_distance) ||
                                    (error == best_error &&
                                    distance == best_distance &&
                                    acc_alpha_reduced > best_acc_alpha_reduced))
                            {
                                best_error = error;
                                best_distance = distance;
                                best_acc_alpha_reduced = acc_alpha_reduced;

                                best_x = x;
                                best_y = y;
                                best_type_idx = type_idx;
                                best_color_idx = color_idx;
                            }
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

private:
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
//        std::array<int,
//                PredefinedBrushes::num_types> threshold_count_per_brush_type{};
        std::array<DType,
                PredefinedBrushes::num_types> acc_alpha_reduced_per_brush_type{};
//        std::array<DType,
//                PredefinedBrushes::num_types> sum_error_if_canvas_per_brush_type{};
        std::array<int,
                PredefinedBrushes::num_types> above_threshold_error_count_per_brush_type{};
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
        using PredefinedBrushes::premul;

        const size_t color_idx{smudge.color_idx};

        for(const auto [dx, dy, alpha, alpha_idx] : smudge.pixels_data()) {
            const auto [x, y] = std::pair{smudge.x + dx, smudge.y + dy};

            if (y < 0 || y >= m_height || x < 0 || x >= m_width) {
                continue;
            }

            const Rgb<DType>& color_premul{premul(alpha_idx, color_idx)};

            func(x, y, color_premul, alpha);
        }
    }

    template <typename Func>
    void for_smudge_pixels_and_neighborhood(
            const Smudge<DType>& smudge,
            Func func)
    {
        using PredefinedBrushes::all_types;

        const SmudgeProperties<DType>& props{all_types[smudge.type_idx]};

        // Doesn't work correctly with brushes bigger than of plus shape (3x3)
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

        for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
            for (size_t alpha_idx = 0; alpha_idx < num_alphas; ++alpha_idx) {
                const Rgb<DType> rgb_error = error_from_reversed_blend(
                        m_rev_canvas(x, y),
                        premul(alpha_idx, color_idx),
                        all_alphas[alpha_idx],
                        m_acc_alphas(x, y)
                );

                m_stats(x, y).errors_per_alpha(alpha_idx, color_idx) = rgb_to_distance(rgb_error);
            }
        }
    }

    void update_distance_per_brush_type(Signed x, Signed y) {
        using PredefinedBrushes::all_types, PredefinedBrushes::all_colors,
                PredefinedBrushes::num_types, PredefinedBrushes::num_colors;

        for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
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

                    sum += color_dist(
                            m_rev_canvas(new_x, new_y),
                            all_colors[color_idx].color
                    ) * m_acc_alphas(new_x, new_y);
                }

                m_stats(x, y).distances_per_brush_type(type_idx, color_idx) = sum;
            }
        }
    }

    void update_errors_if_canvas(Signed x, Signed y) {
        m_abs_errors_if_canvas(x, y) = rgb_to_distance(
                error_from_reversed_blend(
                    m_rev_canvas(x, y),
                    m_canvas(x, y),
                    static_cast<DType>(1),
                    m_acc_alphas(x, y)
        ));
    }

#if 0
    void update_threshold_count_per_brush_type(Signed x, Signed y) {
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

                if (m_acc_alphas(new_x, new_y) > m_threshold_alpha) {
                    ++count;
                }
            }

            m_stats(x, y).threshold_count_per_brush_type[type_idx] = count;
        }
    }
#endif

    void update_errors_per_brush_type(Signed x, Signed y) {
        using PredefinedBrushes::all_types, PredefinedBrushes::num_types,
                PredefinedBrushes::num_colors;

        for (size_t color_idx = 0; color_idx < num_colors; ++color_idx) {
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

                    const size_t alpha_idx{props.alphas_idxs[coord_idx]};

                    sum += m_stats(new_x, new_y).errors_per_alpha(alpha_idx, color_idx);
                }

                m_stats(x, y).errors_per_brush_type(type_idx, color_idx) = sum;
            }
        }
    }

    void update_acc_alpha_reduced_per_brush_type(Signed x, Signed y) {
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

                sum += m_acc_alphas(new_x, new_y) * alpha;
            }

            m_stats(x, y).acc_alpha_reduced_per_brush_type[type_idx] = sum;
        }
    }

#if 0
    void update_sum_error_if_canvas_per_brush_type(Signed x, Signed y) {
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

                sum += m_abs_errors_if_canvas(new_x, new_y);
            }

            m_stats(x, y).sum_error_if_canvas_per_brush_type[type_idx] = sum;
        }
    }
#endif

    void update_threshold_error_count_per_brush_type(Signed x, Signed y) {
        using PredefinedBrushes::all_types, PredefinedBrushes::num_types;

        for (size_t type_idx = 0; type_idx < num_types; ++type_idx) {
            const auto& props{all_types[type_idx]};

            int sum{0};
            for (size_t coord_idx = 0; coord_idx < props.num_pixels; ++coord_idx) {
                const Signed new_x{x + props.xs[coord_idx]};
                const Signed new_y{y + props.ys[coord_idx]};

                if (new_y < 0 || new_y >= m_height ||
                        new_x < 0 || new_x >= m_width) {
                    continue;
                }

                // m_threshold_alpha means "max OK error per pixel" here
                if (m_abs_errors_if_canvas(new_x, new_y) > m_threshold_alpha) {
                    ++sum;
                }
            }

            m_stats(x, y).above_threshold_error_count_per_brush_type[type_idx] = sum;
        }
    }

    void update_errors_added(
            Signed x,
            Signed y,
            const Rgb<DType>& color_premul,
            DType alpha)
    {
        Rgb<DType> error = error_from_reversed_blend(
                m_rev_canvas(x, y), color_premul, alpha, m_acc_alphas(x, y));

        m_errors_added(x, y) += error;
        m_abs_errors_added(x, y) += rgb_to_distance(error);
    }

    void update_rev_canvas_and_acc_alphas(
            Signed x,
            Signed y,
            const Rgb<DType>& color_premul,
            DType alpha)
    {
        m_rev_canvas(x, y) = reverse_blend(
                m_rev_canvas(x, y), color_premul, alpha);

        m_acc_alphas(x, y) *= 1 - alpha;
    }

    void init_pixel_stats() {
        for_all_image([this](Signed x, Signed y) {
            update_errors_per_alpha(x, y);
        });

        for_all_image([this](Signed x, Signed y) {
            update_errors_if_canvas(x, y);
        });

        // Must follow "for all image: update errors per alpha"
        for_all_image([this](Signed x, Signed y) {
            update_errors_per_brush_type(x, y);
        });

        for_all_image([this](Signed x, Signed y) {
            update_distance_per_brush_type(x, y);
        });

#if 0
        for_all_image([this](Signed x, Signed y) {
            update_threshold_count_per_brush_type(x, y);
        });
#endif
        for_all_image([this](Signed x, Signed y) {
            update_acc_alpha_reduced_per_brush_type(x, y);
        });

#if 0
        // Must follow "for all image: update errors if canvas"
        for_all_image([this](Signed x, Signed y) {
            update_sum_error_if_canvas_per_brush_type(x, y);
        });
#endif
        // Must follow "for all image: update errors if canvas"
        for_all_image([this](Signed x, Signed y) {
            update_threshold_error_count_per_brush_type(x, y);
        });
    }

    void reverse_apply_smudge(const Smudge<DType>& smudge) {
        // Must preceide updating of m_acc_alphas 'cause it needs its old values
        for_smudge_pixels(smudge, [this](Signed x, Signed y,
                        const Rgb<DType>& color_premul, DType alpha) {
                    update_errors_added(x, y, color_premul, alpha);
                }
        );

        for_smudge_pixels(smudge, [this](Signed x, Signed y,
                        const Rgb<DType>& color_premul, DType alpha) {
                    update_rev_canvas_and_acc_alphas(x, y, color_premul, alpha);
                }
        );

        // update_errors_added and update_rev_canvas_and_acc_alphas must
        // preceide everything below:

        for_smudge_pixels(smudge, [this](Signed x, Signed y, const Rgb<DType>&, DType) {
                update_errors_per_alpha(x, y);
            });

        for_smudge_pixels(smudge, [this](Signed x, Signed y, const Rgb<DType>&, DType) {
                update_errors_if_canvas(x, y);
            });

        for_smudge_pixels_and_neighborhood(smudge, [this](Signed x, Signed y) {
                update_distance_per_brush_type(x, y);
            });

        // Must follow "for smudge pixels: update errors per alpha"
        for_smudge_pixels_and_neighborhood(smudge, [this](Signed x, Signed y) {
                update_errors_per_brush_type(x, y);
            });

#if 0
        for_smudge_pixels_and_neighborhood(smudge, [this](Signed x, Signed y) {
                update_threshold_count_per_brush_type(x, y);
            });
#endif
        for_smudge_pixels_and_neighborhood(smudge, [this](Signed x, Signed y) {
                update_acc_alpha_reduced_per_brush_type(x, y);
            });

#if 0
        // Must follow "for smudge pixels: update errors if canvas"
        for_smudge_pixels_and_neighborhood(smudge, [this](Signed x, Signed y) {
                update_sum_error_if_canvas_per_brush_type(x, y);
            });
#endif
        // Must follow "for smudge pixels: update errors if canvas"
        for_smudge_pixels_and_neighborhood(smudge, [this](Signed x, Signed y) {
                update_threshold_error_count_per_brush_type(x, y);
            });
    }

    void print_statistics(const std::vector<Smudge<DType>>& result) {
        const double introduced{std::accumulate(m_abs_errors_added.cbegin(),
                m_abs_errors_added.cend(), 0.0)};

        const double from_canvas{std::accumulate(m_abs_errors_if_canvas.cbegin(),
                m_abs_errors_if_canvas.cend(), 0.0)};

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

    const std::vector<size_t> m_allowed_brush_types;

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
