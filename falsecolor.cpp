#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <algorithm>
#include <immintrin.h>
#include <iostream>
#include <memory_resource>
#include <numpy/ndarrayobject.h>
#include <random>
#include <unordered_map>
#include <vector>

// FIXME
#define FC_PYCHECK(w)        \
    if (!(w)) {              \
        PyErr_BadArgument(); \
        return NULL;         \
    }

std::mt19937 init_generator()
{
    std::random_device rd;
    return std::mt19937{rd()};
}

template<typename T>
T random(T start, T end)
{
    static std::mt19937 generator = init_generator();

    std::uniform_int_distribution<T> dist(start, end);
    return dist(generator);
}

struct Rgb {
    float r, g, b;

    Rgb() = default;

    Rgb(float r, float g, float b)
        : r(r)
        , g(g)
        , b(b)
    {
    }

    static Rgb hex(uint8_t r, uint8_t g, uint8_t b)
    {
        Rgb color;

        color.r = float(r) / 255.0f;
        color.g = float(g) / 255.0f;
        color.b = float(b) / 255.0f;

        return color;
    }
};

struct Brush {
    Rgb color;
    std::string name;

    Brush(std::string name, Rgb color)
        : color(color)
        , name(name)
    {
    }
};

static inline __m256 mm256_abs_ps(__m256 x)
{
    return _mm256_andnot_ps(_mm256_set1_ps(-0.f), x);
}

static inline __m256 calc_dist(__m256 a, __m256 b)
{
    __m256 delta = mm256_abs_ps(_mm256_sub_ps(a, b));
    __m256 abs_sq = _mm256_mul_ps(delta, delta);
    __m256 dot_product = _mm256_mul_ps(a, b);
    return _mm256_mul_ps(dot_product, _mm256_mul_ps(abs_sq, abs_sq));
}

#define MAKE_ALIGNED_POINTER(pointer, alignment) ((uintptr_t)(pointer) + (alignment - 1) & ~(uintptr_t)(alignment - 1))

struct Smudge;

class Image {
public:
    Image(size_t width, size_t height, std::pmr::memory_resource* res, const Image* target = nullptr)
        : m_width(width)
        , m_height(height)
        , m_data_size(width * height * 3)
        , m_storage(width * height * 3 + 32, 1.0f, res)
        , m_target(target)
        , m_cached_dist(-1.0f) // default value means not yet calculated distance
    {
        m_data = (float*)MAKE_ALIGNED_POINTER(m_storage.data(), 32);
    }

    Image clone(std::pmr::memory_resource* res) const
    {
        Image im(m_width, m_height, res);
        memcpy(im.m_data, m_data, m_data_size * sizeof(float));

        return im;
    };

    Image(Image&& other) {
        m_width = other.m_width;
        m_height = other.m_height;
        m_data_size = other.m_data_size;
        m_data = other.m_data;
        m_cached_dist = other.m_cached_dist;
        m_storage = std::move(other.m_storage);
    }

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    void set_pixel(size_t x, size_t y, Rgb color)
    {
        if (x >= m_width || y >= m_height) {
            return;
        }

        size_t offset = 3 * (m_width * y + x);

        m_data[offset] = color.r;
        m_data[offset + 1] = color.g;
        m_data[offset + 2] = color.b;
    }

    Rgb get_pixel(size_t x, size_t y) const
    {
        if (x >= m_width || y >= m_height) {
            return Rgb{255, 255, 255};
        }

        size_t offset = 3 * (m_width * y + x);

        Rgb color;

        color.r = m_data[offset];
        color.g = m_data[offset + 1];
        color.b = m_data[offset + 2];

        return color;
    }

    void blend_pixel(size_t x, size_t y, Rgb color, float alpha)
    {
        if (x >= m_width || y >= m_height) {
            return;
        }

        Rgb dst = get_pixel(x, y);

        float r = dst.r * (1.0f - alpha) + color.r * alpha;
        float g = dst.g * (1.0f - alpha) + color.g * alpha;
        float b = dst.b * (1.0f - alpha) + color.b * alpha;

        set_pixel(x, y, Rgb(r, g, b));

        m_cached_dist = dist();
    }

#if 1
    float dist() const
    {
        __m256 out = _mm256_setzero_ps();

        const __m256 const2 = _mm256_set1_ps(2.0f);
        const __m256 const4 = _mm256_set1_ps(4.0f);
        const __m256 const0_5 = _mm256_set1_ps(0.5f);

        __m256 avg_r, dr, dg, db, dr_sqr, dg_sqr, db_sqr, coef_r, coef_g, coef_b, dist_r, dist_g, dist_b;
        std::array<float, 8> r1_arr, g1_arr, b1_arr, r2_arr, g2_arr, b2_arr;
        // Assumption! Image width * height % 8 == 0
        for (size_t i = 0; i < m_data_size; i += 24) {
            for (size_t j = 0; j < 8; j++) {
                r1_arr[j] = m_data[i + 3 * j];
                g1_arr[j] = m_data[i + 3 * j + 1];
                b1_arr[j] = m_data[i + 3 * j + 2];
                r2_arr[j] = m_target->m_data[i + 3 * j];
                g2_arr[j] = m_target->m_data[i + 3 * j + 1];
                b2_arr[j] = m_target->m_data[i + 3 * j + 2];
            }
            __m256 r1 = *((__m256*) r1_arr.data());
            __m256 g1 = *((__m256*) g1_arr.data());
            __m256 b1 = *((__m256*) b1_arr.data());
            __m256 r2 = *((__m256*) r2_arr.data());
            __m256 g2 = *((__m256*) g2_arr.data());
            __m256 b2 = *((__m256*) b2_arr.data());

            avg_r = _mm256_mul_ps(const0_5, _mm256_add_ps(r1, r2));
            dr = _mm256_sub_ps(r1, r2);
            dg = _mm256_sub_ps(g1, g2);
            db = _mm256_sub_ps(b1, b2);
            dr_sqr = _mm256_mul_ps(dr, dr);
            dg_sqr = _mm256_mul_ps(dg, dg);
            db_sqr = _mm256_mul_ps(db, db);
            coef_r = _mm256_add_ps(const2, avg_r);
            coef_g = const4;
            coef_b = _mm256_sub_ps(const2, avg_r);

            dist_r = _mm256_mul_ps(coef_r, dr_sqr);
            dist_g = _mm256_mul_ps(coef_g, dg_sqr);
            dist_b = _mm256_mul_ps(coef_b, db_sqr);
            out = _mm256_add_ps(out, _mm256_mul_ps(dist_r, dist_r));
            out = _mm256_add_ps(out, _mm256_mul_ps(dist_g, dist_g));
            out = _mm256_add_ps(out, _mm256_mul_ps(dist_b, dist_b));
            //out = _mm256_add_ps(out, dist_r);
            //out = _mm256_add_ps(out, dist_g);
            //out = _mm256_add_ps(out, dist_b);
        }

        float ds = 0.0f;
        for (int i = 0; i < 8; i++) {
            ds += ((float*)&out)[i];
        }

        return ds;
    }
#else
    float dist(const Image& other) const
    {
        float d = 0.0f;

        for (size_t i = 0; i < m_data_size; i += 3) {
            float r1 = m_data[i];
            float g1 = m_data[i + 1];
            float b1 = m_data[i + 2];
            float r2 = other.m_data[i];
            float g2 = other.m_data[i + 1];
            float b2 = other.m_data[i + 2];
            float avg_r = (r1 + r2) / 2.0f;
            float dr = fabs(r1 - r2);
            float dg = fabs(g1 - g2);
            float db = fabs(b1 - b2);
            float coef_r = 2.0f + avg_r;
            float coef_g = 4.0f;
            float coef_b = 3.0f - avg_r;
            float delta_r = coef_r * dr;
            float delta_g = coef_g * dg;
            float delta_b = coef_b * db;

            d += r1 * r2 * pow(delta_r, 5) + g1 * g2 * pow(delta_g, 5) + b1 * b2 * pow(delta_b, 5);
        }

        return d;
    }
#endif
    float dist(const std::vector<Smudge>& smudges, const std::vector<Brush>& brushes);

    float dist_diff(size_t x, size_t y, Rgb color, float alpha) const
    {
        if (x < 0 || y < 0 || x >= m_width || y >= m_height) {
            return 0.0f;
        }

        Rgb p1 = get_pixel(x, y);
        Rgb p2 = m_target->get_pixel(x, y);

        float prev_contribution = color_dist(p1, p2);
        Rgb p1_changed = blend_pixels(p1, color, alpha);
        float current_contribution = color_dist(p1_changed, p2);

        return current_contribution - prev_contribution;
    }

    float color_dist(const Rgb& c1, const Rgb& c2) const
    {
        float avg_r = 0.5f * (c1.r + c2.r);
        float dr = c1.r - c2.r;
        float dg = c1.g - c2.g;
        float db = c1.b - c2.b;
        float dr_sqr = dr * dr;
        float dg_sqr = dg * dg;
        float db_sqr = db * db;
        float coef_r = 2.0f + avg_r;
        float coef_g = 4.0f;
        float coef_b = 2.0f - avg_r;
        float dist_r = coef_r * dr_sqr;
        float dist_g = coef_g * dg_sqr;
        float dist_b = coef_b * db_sqr;
        return dist_r * dist_r + dist_g * dist_g + dist_b * dist_b;
    }

    Rgb blend_pixels(const Rgb& base, const Rgb& c, float alpha) const
    {
        float r = base.r * (1.0f - alpha) + c.r * alpha;
        float g = base.g * (1.0f - alpha) + c.g * alpha;
        float b = base.b * (1.0f - alpha) + c.b * alpha;
        return Rgb(r, g, b);
    }

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
    const Image* m_target;
};

struct Smudge {
    size_t x{};
    size_t y{};
    size_t brush_index{};

    Smudge() = default;

    Smudge make_variation(size_t width, size_t height, size_t brush_count) const
    {
        int xdelta = random<int>(-1, 1);
        int ydelta = random<int>(-1, 1);

        Smudge variation;

        variation.x = size_t(std::clamp(int(x) + xdelta, 0, int(width - 1)));
        variation.y = size_t(std::clamp(int(y) + ydelta, 0, int(height - 1)));

        variation.brush_index = brush_index;

        if (random<int>(0, 9) == 0) {
            variation.brush_index = random<size_t>(0, brush_count - 1);
        }

        return variation;
    }

    void apply(Image& image, const std::vector<Brush>& brushes) const
    {
        if (x >= image.width() || y > image.height()) {
            return;
        }

        Rgb color = brushes[brush_index].color;

        image.blend_pixel(x, y, color, 0.7f);
        image.blend_pixel(x, y - 1, color, 0.5f);
        image.blend_pixel(x, y + 1, color, 0.5f);
        image.blend_pixel(x - 1, y, color, 0.5f);
        image.blend_pixel(x + 1, y, color, 0.5f);
    }
};

// Assumption! m_target is not null
float Image::dist(const std::vector<Smudge>& smudges, const std::vector<Brush>& brushes)
{
    if (m_cached_dist == -1.0f) {
        m_cached_dist = dist();
    }

    float distance = m_cached_dist;

    for (auto&& smudge : smudges) {
        if (smudge.x < 0 || smudge.y < 0 || smudge.x >= m_width || smudge.y >= m_height) {
            continue;
        }

        Rgb color = brushes[smudge.brush_index].color;
        distance += dist_diff(smudge.x, smudge.y, color, 0.7f);
        distance += dist_diff(smudge.x, smudge.y - 1, color, 0.5f);
        distance += dist_diff(smudge.x, smudge.y + 1, color, 0.5f);
        distance += dist_diff(smudge.x - 1, smudge.y, color, 0.5f);
        distance += dist_diff(smudge.x + 1, smudge.y, color, 0.5f);
    }

    return distance;
}

struct RankedSmudgePattern;

struct SmudgePattern {
    Smudge smudge1{};
    Smudge smudge2{};
    bool has_smudge2{};

    SmudgePattern() = default;

    static SmudgePattern make_random(size_t width, size_t height, size_t brush_count)
    {
        SmudgePattern pattern;
        pattern.smudge1.x = random<size_t>(0, width - 1);
        pattern.smudge1.y = random<size_t>(0, height - 1);
        pattern.smudge1.brush_index = random<size_t>(0, brush_count - 1);

        if (random<size_t>(0, 1)) {
            pattern.has_smudge2 = true;
            pattern.smudge2 = pattern.smudge1.make_variation(width, height, brush_count);
        }

        return pattern;
    }

    RankedSmudgePattern rank(Image& canvas, const std::vector<Brush>& brushes, std::pmr::memory_resource* res);

    SmudgePattern make_variation(size_t width, size_t height, size_t brush_count) const
    {
        SmudgePattern variation;
        variation.smudge1 = smudge1.make_variation(width, height, brush_count);
        variation.has_smudge2 = has_smudge2;
        if (has_smudge2) {
            variation.smudge2 = smudge2.make_variation(width, height, brush_count);
        }
        return variation;
    }
};

struct RankedSmudgePattern {
    SmudgePattern pattern{};
    float rank{};

    RankedSmudgePattern() = default;
};

RankedSmudgePattern SmudgePattern::rank(Image& canvas, const std::vector<Brush>& brushes, std::pmr::memory_resource* res)
{
    RankedSmudgePattern ranked;

    std::vector<Smudge> temp_smudges = {smudge1};
    if (has_smudge2) {
        temp_smudges.push_back(smudge2);
    }
    const std::vector<Smudge> smudges = temp_smudges;

    ranked.pattern = *this;
    ranked.rank = canvas.dist(smudges, brushes);

    return ranked;
}

std::vector<Smudge> fit_target_image(const Image& target, const std::vector<Brush>& brushes)
{
    Image canvas(target.width(), target.height(), std::pmr::get_default_resource(), &target);

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
            patterns.push_back(SmudgePattern::make_random(canvas.width(), canvas.height(), brushes.size()).rank(canvas, brushes, &pool));
        }

        for (int g = 0; g < generations; g++) {
            std::sort(patterns.begin(), patterns.end(), [=](const RankedSmudgePattern& left, const RankedSmudgePattern& right) {
                return left.rank < right.rank;
            });

            patterns.resize(keep_alive_after_culling);

            for (size_t p = 0; p < keep_alive_after_culling; p++) {
                for (size_t o = 0; o < offspring_count; o++) {
                    auto new_pattern = patterns[p].pattern.make_variation(canvas.width(), canvas.height(), brushes.size()).rank(canvas, brushes, &pool);
                    patterns.push_back(new_pattern);
                }
            }
        }

        std::sort(patterns.begin(), patterns.end(), [=](const RankedSmudgePattern& left, const RankedSmudgePattern& right) {
            return left.rank < right.rank;
        });

        const auto& best_pattern = patterns[0].pattern;

        smudge_history.push_back(best_pattern.smudge1);
        best_pattern.smudge1.apply(canvas, brushes);
        if (best_pattern.has_smudge2) {
            smudge_history.push_back(best_pattern.smudge2);
            best_pattern.smudge2.apply(canvas, brushes);
        }
    }

    return smudge_history;
}

static PyObject* fit(PyObject* self, PyObject* args)
{
    import_array();

    PyObject* input;

    if (!PyArg_ParseTuple(args, "O", &input)) {
        return NULL;
    }

    FC_PYCHECK(PyArray_Check(input));

    PyArrayObject* input_array = ((PyArrayObject*)input);

    // Must be an RGB image without alpha
    FC_PYCHECK(PyArray_DIM(input_array, 2) == 3);
    FC_PYCHECK(PyArray_TYPE(input_array) == NPY_UINT8);

    size_t width = PyArray_DIM(input_array, 0);
    size_t height = PyArray_DIM(input_array, 1);

    Image target(width, height, std::pmr::get_default_resource());

    for (size_t y = 0; y < height; y++) {
        uint8_t* row = (uint8_t*)PyArray_DATA(input_array) + y * PyArray_STRIDES(input_array)[0];

        for (size_t x = 0; x < width; x++) {
            uint8_t r = row[PyArray_STRIDES(input_array)[1] * x + 0 * PyArray_STRIDES(input_array)[2]];
            uint8_t g = row[PyArray_STRIDES(input_array)[1] * x + 1 * PyArray_STRIDES(input_array)[2]];
            uint8_t b = row[PyArray_STRIDES(input_array)[1] * x + 2 * PyArray_STRIDES(input_array)[2]];

            target.set_pixel(x, y, Rgb::hex(r, g, b));
        }
    }

    std::vector<Brush> brushes = {
        {"White", Rgb::hex(0xff, 0xff, 0xff)},
        {"Yellow", Rgb::hex(0xff, 0xf0, 0x00)},
        {"Orange", Rgb::hex(0xff, 0x6c, 0x00)},
        {"Red", Rgb::hex(0xff, 0x00, 0x00)},
        {"Violet", Rgb::hex(0x8a, 0x00, 0xff)},
        {"Blue", Rgb::hex(0x00, 0x0c, 0xff)},
        {"Green", Rgb::hex(0x0c, 0xff, 0x00)},
        {"Magenta", Rgb::hex(0xfc, 0x00, 0xff)},
        {"Cyan", Rgb::hex(0x00, 0xff, 0xea)},
        {"Grey", Rgb::hex(0xbe, 0xbe, 0xbe)},
        {"DarkGrey", Rgb::hex(0x7b, 0x7b, 0x7b)},
        {"Black", Rgb::hex(0x00, 0x00, 0x00)},
        {"DarkGreen", Rgb::hex(0x00, 0x64, 0x00)},
        {"Brown", Rgb::hex(0x96, 0x4b, 0x00)},
        {"Pink", Rgb::hex(0xff, 0xc0, 0xcb)},
    };

    auto steps = fit_target_image(target, brushes);

    PyObject* list = PyList_New(steps.size());

    for (size_t i = 0; i < steps.size(); i++) {
        auto name = brushes[steps[i].brush_index].name;

        PyObject* tuple = Py_BuildValue("(iis#)", int(steps[i].x), int(steps[i].y), name.data(), Py_ssize_t(name.size()));
        PyList_SetItem(list, i, tuple);
    }

    return list;
}

PyMethodDef method_table[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

PyModuleDef falsecolor_module = {
    PyModuleDef_HEAD_INIT,
    "falsecolor",
    "Falsecolor approximates tiny pictures",
    -1,
    method_table,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_falsecolor(void)
{
    return PyModule_Create(&falsecolor_module);
}
