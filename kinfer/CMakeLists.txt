cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(kinfer)

find_package(Torch REQUIRED)

add_executable(segment-image-by-model segment-image-by-model.cpp)
target_link_libraries(segment-image-by-model "${TORCH_LIBRARIES}")
set_property(TARGET segment-image-by-model PROPERTY CXX_STANDARD 14)