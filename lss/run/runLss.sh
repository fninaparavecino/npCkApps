#!/bin/bash
echo "===========LSS non DP=========="
src="../cuda-opt-nodp/lss"
$src --image ../inputs/fractal/fractal.intensities.pgm --params ../inputs/fractal/fractal.params --labels ../inputs/fractal/fractal.label.pgm

echo "==========LSS DP=========="
src="../cuda-opt/lss"
$src --image ../inputs/fractal/fractal.intensities.pgm --params ../inputs/fractal/fractal.params --labels ../inputs/fractal/fractal.label.pgm


