#!/bin/bash
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed,gld_efficiency,gst_efficiency ./selectiveMatrixAddNonNP --size 1024
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed,gld_efficiency,gst_efficiency ./selectiveMatrixAddNonNP --size 2048
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed,gld_efficiency,gst_efficiency ./selectiveMatrixAddNonNP --size 4096
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed,gld_efficiency,gst_efficiency ./selectiveMatrixAddNonNP --size 7168

echo "NP METRICS------------------------------------------------------"
#!/bin/bash
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed,gld_efficiency,gst_efficiency ./selectiveMatrixAddNP --size 1024
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed,gld_efficiency,gst_efficiency ./selectiveMatrixAddNP --size 2048
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed,gld_efficiency,gst_efficiency ./selectiveMatrixAddNP --size 4096
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed,gld_efficiency,gst_efficiency ./selectiveMatrixAddNP --size 7168


