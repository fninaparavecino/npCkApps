#!/bin/bash
echo "===========LSS non NP=========="
src="../optNonNP/lss"
$src --image ../inputs/coins/coins.intensities.pgm --params ../inputs/coins/coins.params --labels ../inputs/coins/coins.label.pgm

echo "==========LSS NP=========="
src="../optNP/lss"
$src --image ../inputs/coins/coins.intensities.pgm --params ../inputs/coins/coins.params --labels ../inputs/coins/coins.label.pgm


