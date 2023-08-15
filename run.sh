#!/bin/sh

cmake . -B build/ -DCAFFE2_USE_CUDNN=1
cd build/
make -j16
./CMakeFiles/poly.dir/main.cpp.o