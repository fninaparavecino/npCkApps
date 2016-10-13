#!/bin/bash
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed ./acclNonNP

echo "NP METRICS------------------------------------------------------"
#!/bin/bash
nvprof --metrics inst_executed,warp_execution_efficiency,cf_executed ./acclNP

