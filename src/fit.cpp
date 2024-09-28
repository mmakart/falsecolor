#include "fit.hpp"
#include "image.hpp"
#include "smudge.hpp"
#include <algorithm>
#include <immintrin.h>

float dist(const Image& target, const Image& canvas, const std::vector<Smudge>& smudges, const std::vector<Brush>& brushes, std::pmr::memory_resource* res)
{
    Image canvas_copy = canvas.clone(res);
    for (auto smudge : smudges) {
        smudge.apply(canvas_copy, target, brushes);
    }

    return canvas_copy.get_dist();
}

struct RankedSmudgePattern {
    SmudgePattern pattern{};
    float rank{};

    RankedSmudgePattern() = default;
};

RankedSmudgePattern rank(const SmudgePattern& pattern, const Image& target, Image& canvas, const std::vector<Brush>& brushes, std::pmr::memory_resource* res)
{
    RankedSmudgePattern ranked;

    std::vector<Smudge> temp_smudges = {pattern.smudge1};
    if (pattern.has_smudge2) {
        temp_smudges.push_back(pattern.smudge2);
    }
    const std::vector<Smudge> smudges = temp_smudges;

    ranked.pattern = pattern;
    ranked.rank = dist(target, canvas, smudges, brushes, res);

    return ranked;
}

std::vector<Smudge> fit_target_image(const Image& target, const std::vector<Brush>& brushes)
{
    Image canvas(target.width(), target.height(), std::pmr::get_default_resource());

    int steps = 256;
    int generations = 128;
    int initial_smudges = 8192;
    int keep_alive_after_culling = 24;
    int offspring_count = 4;
    int linear_chunks = 32;

    std::vector<Smudge> smudge_history;

    std::vector<RankedSmudgePattern> patterns(initial_smudges);

    std::pmr::unsynchronized_pool_resource pool{};

    for (int step = 0; step < steps; step++) {
        patterns.clear();

        for (int i = 0; i < initial_smudges; i++) {
            auto pattern = SmudgePattern::make_random(canvas.width(), canvas.height(), brushes.size());
            patterns.push_back(rank(pattern, target, canvas, brushes, &pool));
        }

        for (int g = 0; g < generations; g++) {
            std::sort(patterns.begin(), patterns.end(), [=](const RankedSmudgePattern& left, const RankedSmudgePattern& right) {
                return left.rank < right.rank;
            });

            patterns.resize(keep_alive_after_culling);

            for (size_t p = 0; p < keep_alive_after_culling; p++) {
                for (size_t o = 0; o < offspring_count; o++) {
                    auto new_pattern = patterns[p].pattern.make_variation(canvas.width(), canvas.height(), brushes.size());
                    patterns.push_back(rank(new_pattern, target, canvas, brushes, &pool));
                }
            }
        }

        std::sort(patterns.begin(), patterns.end(), [=](const RankedSmudgePattern& left, const RankedSmudgePattern& right) {
            return left.rank < right.rank;
        });

        const auto& best_pattern = patterns[0].pattern;

        smudge_history.push_back(best_pattern.smudge1);
        best_pattern.smudge1.apply(canvas, target, brushes);
        if (best_pattern.has_smudge2) {
            smudge_history.push_back(best_pattern.smudge2);
            best_pattern.smudge2.apply(canvas, target, brushes);
        }
    }

    return smudge_history;
}
