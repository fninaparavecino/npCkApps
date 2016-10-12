#!/bin/bash
echo "------------------ Non NP Experiments ------------------------------"
./selectiveMatrixAddNonNP --size 1024 --gpu 3
./selectiveMatrixAddNonNP --size 2048 --gpu 3
./selectiveMatrixAddNonNP --size 4096 --gpu 3
./selectiveMatrixAddNonNP --size 7168 --gpu 3
echo "End ----------------------------------------------------------------"

echo "----------------- NP Experiments -----------------------------------"
./selectiveMatrixAddNP --size 1024 --gpu 3
./selectiveMatrixAddNP --size 2048 --gpu 3
./selectiveMatrixAddNP --size 4096 --gpu 3
./selectiveMatrixAddNP --size 7168 --gpu 3
echo "End ----------------------------------------------------------------"
