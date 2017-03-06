#!/bin/bash
src="../../optNonNP/lss"
echo "=====Gather Metrics====="
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed $src --image ../../inputs/tree/tree.intensities.pgm --labels ../../inputs/tree/tree.label.pgm --params ../../inputs/tree/tree.params
