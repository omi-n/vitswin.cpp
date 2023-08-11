cmake_minimum_required(VERSION 3.23)
project(poly)

set(CMAKE_CXX_STANDARD 17)

set(Torch_DIR thirdparty/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(poly main.cpp models/ConvModel.cpp models/ConvModel.h)
target_link_libraries(poly "${TORCH_LIBRARIES}")