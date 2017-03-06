#!/bin/bash
echo "===========LSS non NP=========="
src="../../optNonNP_GPU/lss"
$src --image ../../inputs/fractal/fractal.intensities.pgm --params ../../inputs/fractal/fractal.params --labels ../../inputs/fractal/fractal.label.pgm

echo "==========LSS NP=========="
src="../../optNP/lss"
$src --image ../../inputs/fractal/fractal.intensities.pgm --params ../../inputs/fractal/fractal.params --labels ../../inputs/fractal/fractal.label.pgm


