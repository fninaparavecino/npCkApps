#!/bin/bash
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed,gld_efficiency,gst_efficiency ./acclNonNP

echo "NP METRICS------------------------------------------------------"
#!/bin/bash
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed,gld_efficiency,gst_efficiency ./acclNP


