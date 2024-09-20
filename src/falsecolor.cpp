#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "fit.hpp"
#include "image.hpp"
#include "random.hpp"
#include "smudge.hpp"
#include <Python.h>
#include <immintrin.h>
#include <memory_resource>
#include <numpy/ndarrayobject.h>
#include <vector>

// FIXME
#define FC_PYCHECK(w)        \
    if (!(w)) {              \
        PyErr_BadArgument(); \
        return NULL;         \
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
