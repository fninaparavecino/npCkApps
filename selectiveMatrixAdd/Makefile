all:
	nvcc selectiveMatrixAddNonNP.cu -o selectiveMatrixAddNonNP -rdc=true -arch=compute_35 -code=compute_35,sm_35 -lcudadevrt
	nvcc selectiveMatrixAddNP.cu -o selectiveMatrixAddNP -rdc=true -arch=compute_35 -code=compute_35,sm_35 -lcudadevrt

clean:
	rm -rf selectiveMatrixAddNonNP selectiveMatrixAddNP
