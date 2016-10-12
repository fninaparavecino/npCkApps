/*******************
 * npCase2: Selective Matrix Addition using nested parallelism. 
 * Author : Fanny Nina-Paravecino
 * Date   : October 2016
 */
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
using namespace std;

__global__ void childKernelSync(int* A, int *B, int *C, int parentIdxVar)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	C[parentIdxVar+idx] = A[parentIdxVar+idx] + B[parentIdxVar+idx];
}
__global__ void parentKernelSync(int* A, int *B, int *C, int *npId, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(A[idx*cols] == 1)
	{
		npId[idx] = idx*cols;
		if (cols > 1024){
			childKernelSync<<<cols/1024, 1024>>>(A, B, C, npId[idx]);
			cudaDeviceSynchronize();
		}
		else{
			childKernelSync<<<1, cols>>>(A, B, C, npId[idx]);
			cudaDeviceSynchronize();
		}
	}
}
__global__ void singleKernel(int* A, int *B, int *C, int rows, int cols)
{
	int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(A[idx*cols] == 1)
	{
		for(int i=0; i < cols; i++)
			C[idx*cols+i] = A[idx*cols+i]+B[idx*cols+i];
	}
}
void printOutput(int *A, int rows, int cols)
{
	for(int i=0; i < rows; i++)
	{
		for(int j=0; j < cols; j++){
			printf("%d ", A[i*cols+j]);
		}
		printf("\n");
	}
}
bool check(int *c1, int *c2, int rows, int cols){
	bool same = true;
	for(int i=0; i < rows; i++)
	{
		for(int j=0; j < cols; j++){
			if(c1[i*cols+j] != c2[i*cols+j]){
				printf("ERROR...[%d %d] ", i, j);
				same = false;
				break;
			}
		}
		if (!same)
			break;
	}
	return same;
}
double getWallTime(){
        struct timeval time;
        if(gettimeofday(&time,NULL)){
                printf("Error getting time\n");
                return 0;
        }
        return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
int getTotalCores(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	int fixCores = 0;
	switch (devProp.major){
		case 2: // Fermi
			if (devProp.minor == 1) cores = mp * 48;
			else cores = mp * 32;
      		break;
     		case 3: // Kepler
      			fixCores = 192;
      			printf("Number of cores per SM:       %d\n", fixCores);
      			cores = mp * fixCores;
      			break;
     		case 5: // Maxwell
      			fixCores = 128;
      			printf("Number of cores per SM:       %d\n", fixCores);
			cores = mp * fixCores;
      			break;
     		case 6: // Pascal
      			if (devProp.minor == 1) {
				fixCores = 128;
      				printf("Number of cores per SM:       %d\n",fixCores);
				cores = mp * fixCores;
			}
      			else if (devProp.minor == 0){
				fixCores = 64;
      				printf("Number of cores per SM:       %d\n",fixCores);
				cores = mp * fixCores;
			}
      			else printf("Unknown device type\n");
      			break;
     		default:
      			printf("Unknown device type\n"); 
      			break;
	}
	return cores;
}
void printDevProp(cudaDeviceProp devProp)
{
    printf("Name:                          %s\n",  devProp.name);
    printf("Capability:                   (%d, %d)\n",  devProp.major, devProp.minor);
    printf("# SM:                          %d\n",  devProp.multiProcessorCount);
    printf("Total Cores:                   %d\n", getTotalCores(devProp));
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("=================================\n");
    return;
}

int main(int argC, char** argV)
{
	
	// Number of CUDA devices
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("CUDA Device Query...\n");
	printf("There are %d CUDA devices.\n", devCount);
	// Iterate through devices
	int gpu = 0;
	cudaDeviceProp devProp;
	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		printf("\nCUDA Device #%d\n", i);
		cudaGetDeviceProperties(&devProp, i);
		printDevProp(devProp);
	}
	cudaSetDevice(gpu);
	///*******************************
	float div = 75.0f;
	int ROWS = 1024, COLS = 1024;
	for(int i=1; i<argC; i=i+2)
	{
		if(strcmp(argV[i], "--size") == 0)
		{
			if(i+1 < argC)
			{
				ROWS = atoi(argV[i+1]);
				COLS = ROWS;
				if(ROWS < 1)
				{
					cerr << "Size must be greater than 0." << endl;
					exit(1);
				}
			}
			else
			{
				printf("Error...\n");
				exit(1);
			}
		}
		else if(strcmp(argV[i], "--div") == 0){
			if(i+1 < argC)
			{
				div = atof(argV[i+1]);
				if(div <= 0)
				{
					cerr << "Divergence must be greater than 0." << endl;
					exit(1);
				}
			}
			else
			{
				printf("Error...\n");
				exit(1);
			}
		}
		else if(strcmp(argV[i], "--gpu") == 0){
                        if(i+1 < argC)
                        {
                                gpu = atoi(argV[i+1]);
                                if(gpu < 0)
                                {
                                        cerr << "GPU index should be a positive index of the array of GPU." << endl;
                                        exit(1);
                                }
                                break;
                        }
                        else
                        {
                                printf("Error...\n");
                                exit(1);
                        }
                }
		else if(strcmp(argV[i], "-h") == 0 || strcmp(argV[i], "--help") == 0)
		{
			cout << "Usage: " << argV[0] << " [OPTIONS] --size <number> --div <number> --gpu <number> " << endl;
			cout << "  -h, --help            Display this information and exit." << endl;

			exit(0);
		}
		else
		{
			cerr << "Did not recognize '" << argV[i] << "'. Try '" << argV[0]
				<< " --help' for additional information." << endl;
			exit(1);
		}
	}

	printf("NP - Characterization: %f percentage of divergence\n", div);
	printf("NP Case2 Matrix Addition: [%d x %d]\n", ROWS, COLS);
	cudaSetDevice(gpu);
	cudaGetDeviceProperties(&devProp, gpu);
	printf("GPU: %s\n", devProp.name);
	int *a = (int*) malloc(ROWS*COLS*sizeof(int));
	int *b = (int*) malloc(ROWS*COLS*sizeof(int));
	int *c = (int*) malloc(ROWS*COLS*sizeof(int));
	int nroChildKernels = 0;
	int rndValue = 0;
	while (nroChildKernels < (ROWS*(div/100.0f))){
		rndValue = rand()%ROWS;
		nroChildKernels++;
		for(int j=0; j<COLS; j++){
			a[rndValue*COLS+j] = 1;
			b[rndValue*COLS+j] = 2;
		}
	}
	printf("Number of child kernels: %d\n", nroChildKernels);
	
	// Sequential
	double wallS0, wallS1;
	wallS0 = getWallTime();
	int *cHost = (int*)malloc(ROWS*COLS*sizeof(int));
	for(int i=0; i<ROWS; i++){
		if(a[i*COLS] == 1)
			for(int j=0; j<COLS; j++){
				cHost[i*COLS+j] = a[i*COLS+j] + b[i*COLS+j];
			}
	}
	wallS1 = getWallTime();
	printf("\tSequential Job Time: %f ms\n", (wallS1-wallS0)*1000);
	
	// Time variables
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
		
	int *devA;
	int *devB;
	cudaMalloc((void**)&devA, ROWS*COLS*sizeof(int));
	cudaMalloc((void**)&devB, ROWS*COLS*sizeof(int));
		
	//Copying [A] and [B] from host memory to device memory.
	cudaMemcpy(devA, a, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
	
	// Grid configuration
	dim3 threads, blocks;	
	if (ROWS >1024){
		threads.x = 1024; threads.y = 1; threads.z = 1;
		blocks.x = ROWS/threads.x; blocks.y = 1; blocks.z = 1;
	}
	else{
		threads.x = ROWS; threads.y = 1; threads.z = 1;
		blocks.x = 1; blocks.y = 1; blocks.z = 1; 
	}
		
	// NP Sync Case ****************************************************************
	int *cNpSync = (int*)malloc(ROWS*COLS*sizeof(int));
	int *npId = (int*)malloc(ROWS*COLS*sizeof(int));
	
	int *devCSync, *devNpId;
	
	cudaMalloc((void**)&devCSync, ROWS*COLS*sizeof(int));
	cudaMalloc((void**)&devNpId, ROWS*COLS*sizeof(int));
	
	cudaMemcpy(devCSync, cNpSync, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devNpId, npId, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaEventRecord(start, 0);
	
	parentKernelSync<<<blocks, threads>>>(devA, devB, devCSync, devNpId, ROWS, COLS);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	//Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel NP Sync Job time: %.2f ms\n", time);
	
	//Retrieve results from device
	cudaMemcpy(cNpSync, devCSync, ROWS*COLS*sizeof(int), cudaMemcpyDeviceToHost);
	
	//Verify correctness	
	check(cNpSync, cHost, ROWS, COLS) ? printf("Results are correct.\n") : printf("Results are not correct.\n");
	
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devCSync);
	cudaFree(devNpId);
}
