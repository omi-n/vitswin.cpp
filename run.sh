#!/bin/sh

cmake . -B build/
cd build/
make -j16
./CMakeFiles/poly.dir/main.cpp.o