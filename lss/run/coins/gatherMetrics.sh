#!/bin/bash
src="../../optNP/lss"
echo "=====Gather Metrics====="
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed $src --image ../../inputs/coins/coins.intensities.pgm --labels ../../inputs/coins/coins.label.pgm --params ../../inputs/coins/coins.params
