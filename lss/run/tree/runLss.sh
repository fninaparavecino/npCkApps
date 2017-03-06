#!/bin/bash
echo "===========LSS non NP=========="
src="../../optNonNP_GPU/lss"
$src --image ../../inputs/tree/tree.intensities.pgm --params ../../inputs/tree/tree.params --labels ../../inputs/tree/tree.label.pgm

echo "==========LSS NP=========="
src="../../optNP/lss"
$src --image ../../inputs/tree/tree.intensities.pgm --params ../../inputs/tree/tree.params --labels ../../inputs/tree/tree.label.pgm


