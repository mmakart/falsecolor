#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "fit.hpp"
#include "image.hpp"
#include "random.hpp"
#include "smudge.hpp"
#include <Python.h>
#include <memory_resource>
#include <numpy/ndarrayobject.h>
#include <vector>

// FIXME
#define FC_PYCHECK(w)        \
    if (!(w)) {              \
        PyErr_BadArgument(); \
        return NULL;         \
    }

template <typename DType>
static void init_pixels(Image& im, PyArrayObject *ndarray)
{
    size_t width = im.width();
    size_t height = im.height();

    for (size_t y = 0; y < height; y++) {
        uint8_t* row = (uint8_t*)PyArray_DATA(ndarray) + y * PyArray_STRIDES(ndarray)[0];

        for (size_t x = 0; x < width; x++) {
            uint8_t r = row[PyArray_STRIDES(ndarray)[1] * x + 0 * PyArray_STRIDES(ndarray)[2]];
            uint8_t g = row[PyArray_STRIDES(ndarray)[1] * x + 1 * PyArray_STRIDES(ndarray)[2]];
            uint8_t b = row[PyArray_STRIDES(ndarray)[1] * x + 2 * PyArray_STRIDES(ndarray)[2]];

            im.set_pixel<DType>(x, y, Rgb<DType>::hex(r, g, b));
        }
    }
}

static PyObject* fit(PyObject* self, PyObject* args)
{
    using DType = float;

    import_array();

    PyObject* target_arg;
    PyObject* canvas_arg;
    double error_tolerance{};

    if (!PyArg_ParseTuple(args, "OOd", &target_arg, &canvas_arg, &error_tolerance)) {
        return NULL;
    }

    FC_PYCHECK(PyArray_Check(target_arg));
    FC_PYCHECK(PyArray_Check(canvas_arg));

    PyArrayObject* target_ndarray = ((PyArrayObject*)target_arg);
    PyArrayObject* canvas_ndarray = ((PyArrayObject*)canvas_arg);

    // Must be an RGB image without alpha
    FC_PYCHECK(PyArray_DIM(target_ndarray, 2) == 3);
    FC_PYCHECK(PyArray_TYPE(target_ndarray) == NPY_UINT8);

    FC_PYCHECK(PyArray_DIM(canvas_ndarray, 2) == 3);
    FC_PYCHECK(PyArray_TYPE(canvas_ndarray) == NPY_UINT8);

    size_t width = PyArray_DIM(target_ndarray, 1);
    size_t height = PyArray_DIM(target_ndarray, 0);

    // Target and canvas must be of equal sizes
    FC_PYCHECK(PyArray_DIM(canvas_ndarray, 1) == width);
    FC_PYCHECK(PyArray_DIM(canvas_ndarray, 0) == height);

    Image target(width, height, std::pmr::get_default_resource());
    Image canvas(width, height, std::pmr::get_default_resource());

    init_pixels<DType>(target, target_ndarray);
    init_pixels<DType>(canvas, canvas_ndarray);

    // TODO: make customizable
    const std::vector<SmudgeProperties<DType>> allowed_types {
        PredefinedBrushes::pixel_props,
        PredefinedBrushes::water_props,
        PredefinedBrushes::oil_props,
    };

    auto steps = fit_target_image<DType>(target, canvas, error_tolerance, allowed_types);

    PyObject* list = PyList_New(steps.size());

    for (size_t i = 0; i < steps.size(); i++) {
        const auto x{steps[i].x};
        const auto y{steps[i].y};
        const auto name{PredefinedBrushes::all_colors[steps[i].color_idx].name};
        const auto type{PredefinedBrushes::all_types[steps[i].type_idx].type};

        PyObject* tuple = Py_BuildValue(
                "(iis#s#)",
                x,
                y,
                name.data(),
                name.length(),
                type.data(),
                type.length()
        );

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
