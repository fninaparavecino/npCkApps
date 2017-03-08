/*
 * Northeastern University
 * Computer Architecture Research Group
 * NUCAR
 */

#include "lss.h"

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}

using namespace std;

void modMaxIter (int value){
	max_iterations = value;
}

__global__ void evolveContour(
		unsigned int* intensity,
		unsigned int* labels,
		signed int* phi,
		signed int* phiOut,
		int gridXSize,
		int gridYSize,
		int* targetLabels,
		int* lowerIntensityBounds,
		int* upperIntensityBounds,
		int max_iterations,
		unsigned int* globalBlockIndicator,
		unsigned int* globalFinishedVariable,
		unsigned int* totalIterations ){
        int tid = threadIdx.x;

	// Total iterations
	totalIterations = &totalIterations[tid];

	// Size in ints
	int size = (gridXSize*gridYSize)<<(BTSB+BTSB);

	// New phi pointer for each label.
	phi    = &phi[tid*size];
	phiOut = &phiOut[tid*size];

	globalBlockIndicator = &globalBlockIndicator[tid*gridXSize*gridYSize];

	// Global synchronization variable
	globalFinishedVariable = &globalFinishedVariable[tid];
	*globalFinishedVariable = 0;

	dim3 dimGrid(gridXSize, gridYSize);
	dim3 dimBlock(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);
	printf("Grid: %d x %d, with block: %d x %d \n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
	// Including border

	unsigned char intensityTilexx, intensityTilexy, intensityTileyx,intensityTileyy; // input
	unsigned char     labelTilexx, labelTilexy, labelTileyx, labelTileyy; // input
	signed char       phiTilexx, phiTilexy, phiTileyx, phiTileyy; // output

	// step 1
	for(int tx= 0; tx < dimBlock.x*dimGrid.x; tx++){
		for(int ty= 0; ty < dimBlock.y*dimGrid.y; ty++){

			volatile int localGBI; // Flags

			//int blockId = by*dimGrid.x+bx;
			// Read Input Data into Shared Memory
			/////////////////////////////////////////////////////////////////////////////////////

			int x = (tx/dimBlock.x)<<BTSB;
			x = x + tx;
			x = x<<TTSB;
			int y = (ty/dimBlock.y)<<BTSB;
			y = y + ty;
			y = y<<TTSB;

			int location = 	(((x>>TTSB)&BTSMask)                ) |
					(((y>>TTSB)&BTSMask) << BTSB        ) |
					((x>>TSB)            << (BTSB+BTSB) ) ;
			location += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;

			int intensityData = intensity[location];
			int     labelData = labels[location];

			// int sharedX = tx*THREAD_TILE_SIZE;
			// int sharedY = ty*THREAD_TILE_SIZE;

			labelTilexx = labelData         & 0xFF;
			labelTilexy = (labelData >>  8) & 0xFF;
			labelTileyx = (labelData >> 16) & 0xFF;
			labelTileyy = (labelData >> 24) & 0xFF;

			intensityTilexx = intensityData         & 0xFF;
			intensityTilexy = (intensityData >>  8) & 0xFF;
			intensityTileyx = (intensityData >> 16) & 0xFF;
			intensityTileyy = (intensityData >> 24) & 0xFF;

			localGBI = 0;

			// Algorithm
			/////////////////////////////////////////////////////////////////////////////////////

			// Initialization
			int ownIntData = intensityTilexx;
			if(ownIntData >= lowerIntensityBounds[tid] &&
			   ownIntData <= lowerIntensityBounds[tid]) {
				if (labelTilexx == targetLabels[tid])
					phiTilexx = 3;
				else {
					localGBI = 1;
					phiTilexx = 1;
				}
			} else {
				if (labelTilexx == targetLabels[tid]){
					phiTilexx = 4;
					localGBI = 1;
				} else {
					phiTilexx = 0;
				}
			}

			ownIntData = intensityTilexy;
			if(ownIntData >= lowerIntensityBounds[tid] &&
			   ownIntData <= lowerIntensityBounds[tid]) {
				if (labelTilexy == targetLabels[tid])
					phiTilexy = 3;
				else {
					localGBI = 1;
					phiTilexy = 1;
				}
			} else {
				if (labelTilexy == targetLabels[tid]){
					phiTilexy = 4;
					localGBI = 1;
				} else {
					phiTilexy = 0;
				}
			}

			ownIntData = intensityTileyx;
			if(ownIntData >= lowerIntensityBounds[tid] &&
			   ownIntData <= lowerIntensityBounds[tid]) {
				if (labelTileyx == targetLabels[tid])
					phiTileyx = 3;
				else {
					localGBI = 1;
					phiTileyx = 1;
				}
			} else {
				if (labelTileyx == targetLabels[tid]){
					phiTileyx = 4;
					localGBI = 1;
				} else {
					phiTileyx = 0;
				}
			}

			ownIntData = intensityTileyy;
			if(ownIntData >= lowerIntensityBounds[tid] &&
			   ownIntData <= lowerIntensityBounds[tid]) {
				if (labelTileyy == targetLabels[tid])
					phiTileyy = 3;
				else {
					localGBI = 1;
					phiTileyy = 1;
				}
			} else {
				if (labelTileyy == targetLabels[tid]){
					phiTileyx = 4;
					localGBI = 1;
				} else {
					phiTileyy = 0;
				}
			}

	    int phiReturnData = phiTilexx        |
					   							(phiTilexy << 8 ) |
					   							(phiTileyx << 16) |
					   							(phiTileyy << 24);
			// int phiReturnData = intensityTilexx        |
			// 		   							(intensityTilexy << 8 ) |
			// 		   							(intensityTileyx << 16) |
			// 		   							(intensityTileyy << 24);

			phi[location] = phiReturnData;
			// phiOut[location] = phiReturnData;
			if (tx == 0 && ty == 0 && localGBI){
				globalBlockIndicator[(ty/dimBlock.x)*dimGrid.x+(tx/dimBlock.x)] = 1;
			}
		}
	}

	int iterations = 0;
	// Including border
	// __shared__ signed char phiTile[TILE_SIZE+2][TILE_SIZE+2+1]; // input/output

	// Flags
	char BlockChange;
	char change;
	int redoBlock;

	do {
		iterations++;
		//step 2
		for(int tx= 0; tx < dimBlock.x * dimGrid.x; tx++){
			for(int ty= 0; ty < dimBlock.y * dimGrid.y; ty++){
				int blockId = (ty/dimBlock.x)*dimGrid.x+(tx/dimBlock.x);

				// Read Global Block Indicator from global memory
				int localGBI = globalBlockIndicator[blockId];

				// Set Block Variables
				redoBlock = 0;

				if (localGBI) {

					// Read Input Data into Shared Memory
					/////////////////////////////////////////////////////////////////////////////////////
					int x = (tx/dimBlock.x)<<BTSB;
					x = x + tx;
					x = x<<TTSB;
					int y = (ty/dimBlock.y)<<BTSB;
					y = y + ty;
					y = y<<TTSB;

					int location = 	(((x>>TTSB)&BTSMask)                ) |
							(((y>>TTSB)&BTSMask) << BTSB        ) |
							((x>>TSB)            << (BTSB+BTSB) ) ;
					location += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;

					int phiData = phi[location];
					int phiDataLeft, phiDataUp;

					// int sharedX = tx*THREAD_TILE_SIZE+1;
					// int sharedY = ty*THREAD_TILE_SIZE+1;

					phiTilexx = phiData         & 0xFF;
					phiTilexy = (phiData >>  8) & 0xFF;
					phiTileyx = (phiData >> 16) & 0xFF;
					phiTileyy = (phiData >> 24) & 0xFF;

					//Gather neighbors
					if (tx == 0 && ty ==0){
						phiDataLeft = 0;
						phiDataUp = 0;
					}
					else if (tx == 0){
						phiDataUp = 0;
						y = ((ty-1)/dimBlock.y)<<BTSB;
						y = y + ty;
						y = y<<TTSB;

						int locationLeft = 	(((x>>TTSB)&BTSMask)                ) |
								(((y>>TTSB)&BTSMask) << BTSB        ) |
								((x>>TSB)            << (BTSB+BTSB) ) ;
						locationLeft += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
						phiDataLeft = phi[locationLeft];
					}
					else if(ty == 0){
						phiDataLeft = 0;
						x = ((tx-1)/dimBlock.x)<<BTSB;
						x = x + tx;
						x = x<<TTSB;

						int locationUp = 	(((x>>TTSB)&BTSMask)                ) |
								(((y>>TTSB)&BTSMask) << BTSB        ) |
								((x>>TSB)            << (BTSB+BTSB) ) ;
						locationUp += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
						phiDataUp = phi[locationUp];
					}
					else{
						x = ((tx-1)/dimBlock.x)<<BTSB;
						x = x + tx;
						x = x<<TTSB;

						int locationUp = 	(((x>>TTSB)&BTSMask)                ) |
								(((y>>TTSB)&BTSMask) << BTSB        ) |
								((x>>TSB)            << (BTSB+BTSB) ) ;
						locationUp += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
						phiDataUp = phi[locationUp];

						x = (tx/dimBlock.x)<<BTSB;
						x = x + tx;
						x = x<<TTSB;

						y = ((ty-1)/dimBlock.y)<<BTSB;
						y = y + ty;
						y = y<<TTSB;

						int locationLeft = 	(((x>>TTSB)&BTSMask)                ) |
								(((y>>TTSB)&BTSMask) << BTSB        ) |
								((x>>TSB)            << (BTSB+BTSB) ) ;
						locationLeft += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
						phiDataLeft = phi[locationLeft];
					}

					// Algorithm
					/////////////////////////////////////////////////////////////////////
					BlockChange = 0; // Shared variable
					change      = 1; // Shared variable
					// unsigned char phiTileUpxx = phiDataUp         & 0xFF;
					// unsigned char phiTileUpxy = (phiDataUp >>  8) & 0xFF;
					unsigned char phiTileUpyx = (phiDataUp >> 16) & 0xFF;
					// unsigned char phiTileUpyy = (phiDataUp >> 24) & 0xFF;

					// unsigned char phiTileLeftxx = phiDataLeft         & 0xFF;
					unsigned char phiTileLeftxy = (phiDataLeft >>  8) & 0xFF;
					// unsigned char phiTileLeftyx = (phiDataLeft >> 16) & 0xFF;
					// unsigned char phiTileLeftyy = (phiDataLeft >> 24) & 0xFF;

					while (change){
							change = 0;

							if( phiTilexx  == 1 &&
								 (phiTileUpyx  == 3 ||
									phiTilexy  == 3 ||
									phiTileyx  == 3 ||
									phiTileLeftxy  == 3 )){
								phiTilexx = 3;
								change = 1;
								BlockChange = 1;
							} else if ( phiTilexx  == 4 &&
								 (phiTileUpyx  == 0 ||
									phiTilexy  == 0 ||
									phiTileyx  == 0 ||
									phiTileLeftxy  == 0 )){
								phiTilexx = 0;
								change = 1;
								BlockChange = 1;
							}
					}


					if (BlockChange){

						char phiData1 = phiTilexx;
						char phiData2 = phiTilexy;
						char phiData3 = phiTileyx;
						char phiData4 = phiTileyy;

						if (phiData1 ==  4 || phiData2 ==  4 || phiData3 ==  4 || phiData4 ==  4 ||
								phiData1 == 1 || phiData2 == 1 || phiData3 == 1 || phiData4 == 1){
							redoBlock = 1;
						}

						int phiReturnData = phiData1        |
									 (phiData2 << 8 ) |
									 (phiData3 << 16) |
									 (phiData4 << 24);

						phi[location] = phiReturnData;

						if (tx == 0 && ty == 0) {
							if (!redoBlock){
								globalBlockIndicator[blockId] = 0;
							}
							*globalFinishedVariable = 1;
						}
					}
				}
			}
		}
	} while (atomicExch(globalFinishedVariable,0) && (iterations < max_iterations));

	//step 3
	// Including border

	for(int tx= 0; tx < dimBlock.x*dimGrid.x; tx++){
		for(int ty= 0; ty < dimBlock.y*dimGrid.y; ty++){
					// Read Input Data into Shared Memory
					/////////////////////////////////////////////////////////////////////////////////////
					int x = (tx/dimBlock.x)<<BTSB;
					x = x + tx;
					x = x<<TTSB;
					int y = (ty/dimBlock.y)<<BTSB;
					y = y + ty;
					y = y<<TTSB;

					int location = 	(((x>>TTSB)&BTSMask)                ) |
							(((y>>TTSB)&BTSMask) << BTSB        ) |
							((x>>TSB)            << (BTSB+BTSB) ) ;
					location += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;

					phiOut[location] = phi[location];
		}
	}

	*totalIterations = iterations;
}

unsigned char *levelSetSegment(
		unsigned char *intensity,
		unsigned char *labels,
		int height,
		int width,
		int *targetLabels,
		int *lowerIntensityBounds,
		int *upperIntensityBounds,
		int numLabels){

	#if defined(DEBUG)
		printf("Printing input data\n");
		printf("Height: %d\n", height);
		printf("Width: %d\n", width);
		printf("Num Labels: %d\n", numLabels);

		for (int i = 0; i < numLabels; i++){
			printf("target label: %d\n", targetLabels[i]);
			printf("lower bound: %d\n", lowerIntensityBounds[i]);
			printf("upper bound: %d\n", upperIntensityBounds[i]);
		}
	#endif

	// int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	// int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	//by FNP
	int gridXSize = ((width/4 + TILE_SIZE -1) / TILE_SIZE);
	int gridYSize = ((height/4 + TILE_SIZE-1) / TILE_SIZE);

	#if defined(DEBUG)
		printf("\n Grid Size: %d %d\n", gridYSize, gridXSize);
		printf(  "Block Size: %d %d\n", BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);
	#endif

	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;

	// Both are the same size (CPU/GPU).
	gpu.size = XSize*YSize*sizeof(char);

	// Allocate arrays in GPU memory
	#if defined(VERBOSE)
		printf ("Allocating arrays in GPU memory.\n");
	#endif

	#if defined(CUDA_TIMING)
		float Ttime;
		TIMER_CREATE(Ttime);
		TIMER_START(Ttime);
	#endif

	checkCuda(cudaMalloc((void**)&gpu.targetLabels           , numLabels*sizeof(int)));
	checkCuda(cudaMalloc((void**)&gpu.lowerIntensityBounds   , numLabels*sizeof(int)));
  checkCuda(cudaMalloc((void**)&gpu.upperIntensityBounds   , numLabels*sizeof(int)));
	checkCuda(cudaMalloc((void**)&gpu.intensity              , gpu.size));
	checkCuda(cudaMalloc((void**)&gpu.labels                 , gpu.size));
	checkCuda(cudaMalloc((void**)&gpu.phi                    , numLabels*gpu.size));
	checkCuda(cudaMalloc((void**)&gpu.phiOut                 , numLabels*gpu.size));
	checkCuda(cudaMalloc((void**)&gpu.globalBlockIndicator   , numLabels*gridXSize*gridYSize*sizeof(int)));
	checkCuda(cudaMalloc((void**)&gpu.globalFinishedVariable , numLabels*sizeof(int)));
	checkCuda(cudaMalloc((void**)&gpu.totalIterations        , numLabels*sizeof(int)));

	// Allocate result array in CPU memory
	gpu.phiOnCpu = new unsigned char[gpu.size*numLabels];
	gpu.totalIterationsOnCpu = new unsigned int [numLabels];

        checkCuda(cudaMemcpy(
			gpu.targetLabels,
			targetLabels,
			numLabels*sizeof(int),
			cudaMemcpyHostToDevice));

        checkCuda(cudaMemcpy(
			gpu.lowerIntensityBounds,
			lowerIntensityBounds,
			numLabels*sizeof(int),
			cudaMemcpyHostToDevice));

        checkCuda(cudaMemcpy(
			gpu.upperIntensityBounds,
			upperIntensityBounds,
			numLabels*sizeof(int),
			cudaMemcpyHostToDevice));

        checkCuda(cudaMemcpy(
			gpu.intensity,
			intensity,
			gpu.size,
			cudaMemcpyHostToDevice));

        checkCuda(cudaMemcpy(
			gpu.labels,
			labels,
			gpu.size,
			cudaMemcpyHostToDevice));

	#if defined(KERNEL_TIMING)
		checkCuda(cudaDeviceSynchronize());
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif

	#if defined(VERBOSE)
		printf("Running algorithm on GPU.\n");
	#endif

	// Launch kernel to begin image segmenation
	evolveContour<<<1, numLabels>>>((unsigned int*)gpu.intensity,
					(unsigned int*)gpu.labels,
					(signed int*)gpu.phi,
					(signed int*)gpu.phiOut,
					gridXSize,
					gridYSize,
					gpu.targetLabels,
					gpu.lowerIntensityBounds,
					gpu.upperIntensityBounds,
					max_iterations,
					gpu.globalBlockIndicator,
					gpu.globalFinishedVariable,
					gpu.totalIterations);


	#if defined(KERNEL_TIMING)
		checkCuda(cudaDeviceSynchronize());
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif

	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(
			gpu.phiOnCpu,
			gpu.phiOut,
			numLabels*gpu.size,
			cudaMemcpyDeviceToHost));

	checkCuda(cudaMemcpy(
			gpu.totalIterationsOnCpu,
			gpu.totalIterations,
			numLabels*sizeof(int),
			cudaMemcpyDeviceToHost));

	// Free resources and end the program
	checkCuda(cudaFree(gpu.intensity));
	checkCuda(cudaFree(gpu.labels));
	checkCuda(cudaFree(gpu.phi));
	checkCuda(cudaFree(gpu.phiOut));
	checkCuda(cudaFree(gpu.targetLabels));
	checkCuda(cudaFree(gpu.lowerIntensityBounds));
	checkCuda(cudaFree(gpu.upperIntensityBounds));
	checkCuda(cudaFree(gpu.globalBlockIndicator));
	checkCuda(cudaFree(gpu.globalFinishedVariable));

	#if defined(CUDA_TIMING)
		TIMER_END(Ttime);
		printf("Total GPU Execution Time: %f ms\n", Ttime);
	#endif

	#if defined(VERBOSE)
		for (int i = 0; i < numLabels; i++){
			printf("target label: %d converged in %d iterations.\n",
					targetLabels[i],
					gpu.totalIterationsOnCpu[i]);
		}
	#endif

	return(gpu.phiOnCpu);

}
