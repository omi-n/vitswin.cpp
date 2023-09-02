cmake_minimum_required(VERSION 3.22)
project(poly)

set(CMAKE_CXX_STANDARD 17)

# set(CAFFE2_USE_CUDNN 1)
set(Torch_DIR thirdparty/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE SRC_FILES models/*.cpp models/*.h)

add_compile_options(-fsanitize=address,undefined -Wall -Wextra)
add_link_options(-fsanitize=address,undefined)

add_executable(poly main.cpp ${SRC_FILES})
target_link_libraries(poly "${TORCH_LIBRARIES}")