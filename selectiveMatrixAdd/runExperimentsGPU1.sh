#!/bin/bash
echo "------------------ Non NP Experiments ------------------------------"
./selectiveMatrixAddNonNP --size 1024 --gpu 1
./selectiveMatrixAddNonNP --size 2048 --gpu 1
./selectiveMatrixAddNonNP --size 4096 --gpu 1
./selectiveMatrixAddNonNP --size 7168 --gpu 1
echo "End ----------------------------------------------------------------"

echo "----------------- NP Experiments -----------------------------------"
./selectiveMatrixAddNP --size 1024 --gpu 1
./selectiveMatrixAddNP --size 2048 --gpu 1
./selectiveMatrixAddNP --size 4096 --gpu 1
./selectiveMatrixAddNP --size 7168 --gpu 1
echo "End ----------------------------------------------------------------"
