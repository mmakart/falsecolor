cmake_minimum_required(VERSION 3.14)
project(falsecolor)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

add_library(falsecolor SHARED ${CMAKE_CURRENT_LIST_DIR}/falsecolor.cpp)

if(NOT Python3_NumPy_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

target_compile_features(falsecolor PRIVATE cxx_std_17)
target_include_directories(falsecolor PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(falsecolor PRIVATE Python3::Module)
if(NOT MSVC)
    target_compile_options(falsecolor PUBLIC -mavx)
endif()

set_target_properties(falsecolor PROPERTIES
    PREFIX ""
    OUTPUT_NAME "falsecolor"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    LINKER_LANGUAGE C
)

if (WIN32)
    set_target_properties(falsecolor PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(falsecolor PROPERTIES SUFFIX ".so")
endif()
