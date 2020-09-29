#!/bin/bash

git clone https://github.com/kokkos/kokkos.git
cd kokkos
git checkout develop
git checkout c836655e8236a57cf1
cd ..
# set up path to nvcc_wrapper
export PATH=$PATH:`pwd`/kokkos/bin
git clone https://github.com/kokkos/kokkos-kernels.git
cd kokkos-kernels
git checkout MergeEntriesReproducer
cd ..
mkdir build
cd build
cp ../kokkos-kernels/config.sh .
./config.sh
make -j16
cp ../kokkos-kernels/A_18.mtx perf_test/sparse
cp ../kokkos-kernels/B_18.mtx perf_test/sparse
cd perf_test/sparse
./sparse_spadd --cuda 0 --amtx ~/A_18.mtx --bmtx ~/B_18.mtx --repeat 1000
