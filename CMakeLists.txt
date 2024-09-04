cmake_minimum_required(VERSION 3.10)
project(falsecolor)

find_package(PythonInterp 3.8 REQUIRED)
find_package(PythonLibs 3.8 REQUIRED)

add_library(falsecolor SHARED ${CMAKE_CURRENT_LIST_DIR}/falsecolor.cpp)

exec_program(${PYTHON_EXECUTABLE} ARGS "-c \"import numpy; print(numpy.get_include())\""
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RETURN_VALUE NUMPY_NOT_FOUND)

if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

target_compile_features(falsecolor PRIVATE cxx_std_17)
target_include_directories(falsecolor PRIVATE ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})
target_link_libraries(falsecolor PRIVATE ${PYTHON_LIBRARY})

set_target_properties(falsecolor PROPERTIES
    PREFIX ""
    OUTPUT_NAME "falsecolor"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    LINKER_LANGUAGE C
)

if (WIN32)
    set_target_properties(falsecolor PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(falsecolor PROPERTIES SUFFIX ".so")
endif()
