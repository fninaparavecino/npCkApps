#!/bin/bash
echo "Non NP Experiments ------------------------------"
./selectiveMatrixAddNonNP --size 1024
./selectiveMatrixAddNonNP --size 2048
./selectiveMatrixAddNonNP --size 4096
./selectiveMatrixAddNonNP --size 8192
echo "End ------------------------------------------------------"

echo "Non NP Experiments ------------------------------"
./selectiveMatrixAddNP --size 1024
./selectiveMatrixAddNP --size 2048
./selectiveMatrixAddNP --size 4096
./selectiveMatrixAddNP --size 8192
echo "End --------------------------------------------------"
