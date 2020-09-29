#!/bin/bash

rm CMakeCache.txt
rm -rf CMakeFiles
rm -rf kokkos-install

../kokkos-kernels/cm_generate_makefile.bash \
    --with-devices=serial,cuda \
    --with-ordinals=int64_t --with-offsets=size_t \
    --arch=SKX,Volta70 --release \
    --compiler=nvcc_wrapper
