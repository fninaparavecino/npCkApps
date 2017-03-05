#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <limits.h>
#include <cstring>
#include <cmath>

#define THREADSX 16
#define THREADSY 16
#define THREADS 512
#define COLS 512
#define COLSHALF 256
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define BUF_SIZE 256
#define MAX_LABELS 262144

class errorHandler { };
using namespace std;

typedef unsigned char uchar;

typedef struct { uchar r, g, b; } rgb;

typedef struct { int x; int y; float jth; } threshold;

inline bool operator==(const rgb &a, const rgb &b)
{
	return ((a.r == b.r) && (a.g == b.g) && (a.b == b.b));
}

template <class T>
inline T abs(const T &x) { return (x > 0 ? x : -x); };

template <class T>
inline int sign(const T &x) { return (x >= 0 ? 1 : -1); };

template <class T>
inline T square(const T &x) { return x*x; };

template <class T>
inline T bound(const T &x, const T &min, const T &max)
{
  return (x < min ? min : (x > max ? max : x));
}

template <class T>
inline bool check_bound(const T &x, const T&min, const T &max)
{
  return ((x < min) || (x > max));
}

template <class T>
class image {
 public:
  /* create an image */
  image(const int width, const int height, const bool init = true);

  /* delete an image */
  ~image();

  /* init an image */
  void init(const T &val);

  /* copy an image */
  image<T> *copy() const;

  /* get the width of an image. */
  int width() const { return w; }

  /* get the height of an . */
  int height() const { return h; }

  /* image data. */
  T *data;

  /* row pointers. */
  T **access;

 private:
  int w, h;
};

/* use imRef to access image data. */
#define imRef(im, x, y) (im->access[y][x])

/* use imPtr to get pointer to image data. */
#define imPtr(im, x, y) &(im->access[y][x])
template <class T>
image<T>::image(const int width, const int height, const bool init)
{
  w = width;
  h = height;
  data = new T[w * h];  // allocate space for image data
  access = new T*[h];   // allocate space for row pointers

  // initialize row pointers
  for (int i = 0; i < h; i++)
    access[i] = data + (i * w);

  if (init)
    memset(data, 0, w * h * sizeof(T));
}

template <class T>
image<T>::~image()
{
  delete [] data;
  delete [] access;
}

template <class T>
void image<T>::init(const T &val)
{
  T *ptr = imPtr(this, 0, 0);
  T *end = imPtr(this, w-1, h-1);
  while (ptr <= end)
    *ptr++ = val;
}

template <class T>
image<T> *image<T>::copy() const
{
  image<T> *im = new image<T>(w, h, false);
  memcpy(im->data, data, w * h * sizeof(T));
  return im;
}

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA kernels
__global__ void findSpansKernel(int *out, int *components, const int *in,
                                const int rows, const int cols)
{
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint colsSpans = ((cols+2-1)/2)*2;
    int current;
    int colsComponents = colsSpans/2;
	bool flagFirst = true;
	int indexOut = 0;
    int indexComp = 0;
    int comp = i*colsComponents;
    if (i<rows)
    {
        for (int j = 0; j < cols; j++)
        {
            if(flagFirst && in[i*cols+j]> 0)
            {
                current = in[i*cols+j];
                out[i*colsSpans+indexOut] = j;
                indexOut++;
                flagFirst = false;
            }
            if (!flagFirst && in[i*cols+j] != current)
            {
                out[i*colsSpans+indexOut] = j-1;
                indexOut++;
                flagFirst = true;
                /*add the respective label*/
                components[i*colsComponents+indexComp] = comp;
                indexComp++;
                comp++;
            }
        }
        if (!flagFirst)
        {
            out[i*colsSpans+indexOut] = cols - 1;
            /*add the respective label*/
            components[i*colsComponents+indexComp] = comp;
        }
    }
}

__global__ void relabelKernel(int *components, int previousLabel, int newLabel, const int colsComponents)
{
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if(components[i*colsComponents+j]==previousLabel)
    {
        components[i*colsComponents+j] = newLabel;
    }
}

__global__ void relabel2Kernel(int *components, int previousLabel, int newLabel, const int colsComponents, const int idx, const int frameRows)
{
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    i = i*colsComponents+j;
    i = i +(colsComponents*frameRows*idx);
    if(components[i]==previousLabel)
    {
        components[i] = newLabel;
    }

}
__global__ void relabelUnrollKernel(int *components, int previousLabel, int newLabel, const int colsComponents, const int idx, const int frameRows, const int factor)
{
    uint id_i_child = (blockIdx.x * blockDim.x) + threadIdx.x;
    id_i_child = id_i_child +(frameRows*idx);
    uint id_j_child = (blockIdx.y * blockDim.y) + threadIdx.y;
    id_j_child  = (colsComponents/factor)*id_j_child;
    uint i = id_i_child;
    for (int j=id_j_child; j< (colsComponents/factor); j++)
    {
        if(components[i*colsComponents+j]==previousLabel)
        {
            components[i*colsComponents+j] = newLabel;
        }
    }
}
__global__ void mergeSpansKernel(int *components, int *spans, const int rows, const int cols, const int frameRows)
{
    uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint colsSpans = ((cols+2-1)/2)*2;
    uint colsComponents = colsSpans/2;
    /*Merge Spans*/
    int startX, endX, newStartX, newEndX;
    int label=-1;
    /*threads and blocks need to relabel the components labels*/
    int threads = 16;
    const int factor =4;

    /*--------For 256, 512--------*/
    dim3 threadsPerBlockUnrollRelabel(threads*threads);
    dim3 numBlocksUnrollRelabel((frameRows*factor)/(threads*threads));
    /*-----------------*/

    for (int i = idx*frameRows; i < ((idx*frameRows)+frameRows)-1; i++) //compute until penultimate row, since we need the below row to compare
    {
        for (int j=0; j < colsSpans/4 && spans[i*colsSpans+j] >=0; j=j+2) //verify if there is a Span available
        {
            startX = spans[i*colsSpans+j];
            endX = spans[i*colsSpans+j+1];
            int newI = i+1; //line below
            for (int k=0; k<colsSpans/4 && spans[newI*colsSpans+k] >=0; k=k+2) //verify if there is a New Span available
            {
                newStartX = spans[newI*colsSpans+k];
                newEndX = spans[newI*colsSpans+k+1];
                if (startX <= newEndX && endX >= newStartX)//Merge components
                {
                    label = components[i*(colsSpans/2)+(j/2)];          //choose the startSpan label
                    
                    // Relabel
//                    __syncthreads();
//                    for (int q = idx*frameRows; q <= i+1; q++)
//					{
//						for (int r=0; r < k/2; r++)
//						{
//							if(components[q*colsComponents+r]==components[newI*(colsSpans/2)+(k/2)])
//							{
//								components[q*colsComponents+r] = label;
//								__syncthreads();
//							}
//						}
//					}
                    for (int p = idx*frameRows; p <= newI; p++)                          /*relabel*/
					{
//						for(int q = 0; q < colsComponents/128; q=q+8)
//						{
							if(components[p*colsComponents]==components[newI*colsComponents+(k/2)])
							{
								components[p*colsComponents] = label;
							}
							if(components[p*colsComponents+1]==components[newI*colsComponents+(k/2)])
							{
								components[p*colsComponents+1] = label;
							}
							if(components[p*colsComponents+2]==components[newI*colsComponents+(k/2)])
							{
								components[p*colsComponents+2] = label;
							}
							if(components[p*colsComponents+3]==components[newI*colsComponents+(k/2)])
							{
								components[p*colsComponents+3] = label;
							}
							
							if(components[p*colsComponents+4]==components[newI*colsComponents+(k/2)])
							{
								components[p*colsComponents+4] = label;
							}
							if(components[p*colsComponents+5]==components[newI*colsComponents+(k/2)])
							{
								components[p*colsComponents+5] = label;
							}
							if(components[p*colsComponents+6]==components[newI*colsComponents+(k/2)])
							{
								components[p*colsComponents+6] = label;
							}
							if(components[p*colsComponents+7]==components[newI*colsComponents+(k/2)])
							{
								components[p*colsComponents+7] = label;
							}
						//}
					}
//                    relabelUnrollKernel<<<numBlocksUnrollRelabel, threadsPerBlockUnrollRelabel>>>(components, components[newI*(colsSpans/2)+(k/2)], label, colsComponents, idx, frameRows, factor);

                }
            }
        }
    }
}

void acclCuda(int *out, int *components, const int *in, const uint nFrames,
                 const int rows, const int cols)
{
    int *devIn = 0;
    int *devComponents = 0;
    int *devOut = 0;

    const int colsSpans = ((cols+2-1)/2)*2; /*ceil(cols/2)*2*/
    const int colsComponents = colsSpans/2;

    /*compute sizes of matrices*/
    const int sizeIn = rows * cols;    
    const int sizeComponents = colsComponents*rows;
    const int sizeOut = colsSpans*rows;

    /*Block and Grid size*/
    int blockSize;
    int gridSize;

    /*Frame Info*/
    const int frameRows = rows/nFrames;

    /*Streams Information*/    
    uint nFramesPerStream = 2;
    uint nStreams = nFrames/nFramesPerStream;

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Choose which GPU to run on, change this on a multi-GPU system.*/
    cudaErrChk(cudaSetDevice(0));

    /* Allocate GPU buffers for three vectors (two input, one output)*/
    cudaErrChk(cudaMalloc((void**)&devOut, sizeOut * sizeof(int)));
    cudaErrChk(cudaMalloc((void**)&devComponents, sizeComponents * sizeof(int)));    
    cudaErrChk(cudaMalloc((void**)&devIn, sizeIn * sizeof(int)));
	
    /* Copy input vectors from host memory to GPU buffers*/
    cudaErrChk(cudaMemcpy(devIn, in, sizeIn * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(devComponents, components, sizeComponents * sizeof(int),
                          cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(devOut, out, sizeOut * sizeof(int), cudaMemcpyHostToDevice));

    /*variables for streaming*/
    const int frameSpansSize = rows/nStreams * colsSpans;
    const int frameCompSize = rows/nStreams * colsComponents;

    /* Round up according to array size */
    blockSize = 1024;
    gridSize = (rows/nStreams)/blockSize;
    //gridSize = rows/blockSize;

    /* Launch a kernel on the GPU with one thread for each element*/
    printf("Number of frames processed: %d\n", nFrames);
    printf("Number of streams created: %d\n", nStreams);
    printf("Grid Configuration findSpans blocks: %d and threadsPerBlock:%d \n", gridSize, blockSize);
    printf("Grid Configuration MergeSpans blocks: %d and threadsPerBlock:%d \n", 1, nFramesPerStream);
    cudaEventRecord(start, 0);      /*measure time*/
    for(int i=0; i<nStreams; ++i)
    {
        findSpansKernel<<<gridSize, blockSize>>>(&devOut[i*frameSpansSize],
                &devComponents[i*frameCompSize], &devIn[i*frameSpansSize],
                rows, cols);
        mergeSpansKernel<<<1, nFramesPerStream>>>(&devComponents[i*frameCompSize],
                                                 &devOut[i*frameSpansSize],
                                                 rows,
                                                 cols,
                                                 frameRows);
    }
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf ("Time kernel execution: %f ms\n", time);
	
    /* Copy device to host*/
    cudaErrChk(cudaMemcpy(components, devComponents, sizeComponents * sizeof(int),
                          cudaMemcpyDeviceToHost));
    cudaErrChk(cudaMemcpy(out, devOut, sizeOut * sizeof(int),
                          cudaMemcpyDeviceToHost));

    /*Free*/
    cudaFree(devOut);
    cudaFree(devIn);
    cudaFree(devComponents);
}

/*
 * RGB generation colors randomly
 */
rgb randomRgb()
{
    rgb c;

    c.r = (uchar)rand();
    c.g = (uchar)rand();
    c.b = (uchar)rand();
    return c;
}


/*
 * getWallTime: Compute timing of execution including I/O
 */
double getWallTime()
{
    struct timeval time;
    if (gettimeofday(&time,NULL))
    {
        printf("Error getting time\n");
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

/*
 * getCpuTime: Compute timing of execution using Clocks function from C++
 */
double getCpuTime()
{
    return (double)clock() / CLOCKS_PER_SEC;
}

void acclSerial(int *imInt, int *spans, int *components, const int rows,
                const int cols, image<rgb> *output)
{
    const int width = cols;
    const int height = rows;
    const int rowsSpans = rows;
    const int colsSpans = ((cols+2-1)/2)*2; //ceil(cols/2)*2
    const int spansSize = colsSpans*rowsSpans;
    const int componentsSize = (colsSpans/2)*rowsSpans;
    int colsComponents = colsSpans/2;
    memset(spans, -1, spansSize*sizeof(int));
    memset(components, -1, componentsSize*sizeof(int));

    /*
     * Find Spans
     */
    double wall0 = getWallTime();
    double cpu0  = getCpuTime();
    for(int i=0; i<rows-1; i++)
    {
        int current =-1;
        bool flagFirst = true;
        int indexOut = 0;
        int indexComp = 0;
        int comp = i*colsComponents;
        for (int j = 0; j < cols; j++)
        {
            if(flagFirst && imInt[i*cols+j]> 0)
            {
                current = imInt[i*cols+j];
                spans[i*colsSpans+indexOut] = j;
                indexOut++;
                flagFirst = false;
            }
            if (!flagFirst && imInt[i*cols+j] != current)
            {
                spans[i*colsSpans+indexOut] = j-1;
                indexOut++;
                flagFirst = true;                
                components[i*colsComponents+indexComp] = comp;  /*Add respective label*/
                indexComp++;
                comp++;
            }
        }
        if (!flagFirst)
        {
            spans[i*colsSpans+indexOut] = cols - 1;
            /*Add the respective label*/
            components[i*colsComponents+indexComp] = comp;
        }
    }

 /*
     * Merge Spans
     */
    int label = -1;
    int startX, endX, newStartX, newEndX;
    for (int i = 0; i < rowsSpans-1; i++) /*compute until penultimate row, since we need the below row to compare*/
    {
        for (int j=0; j < colsSpans-1 && spans[i*colsSpans+j] >=0; j=j+2) /*verify if there is a Span available*/
        {
            startX = spans[i*colsSpans+j];
            endX = spans[i*colsSpans+j+1];
            int newI = i+1; /*line below*/
            for (int k=0; k<colsSpans-1 && spans[newI*colsSpans+k] >=0; k=k+2) /*verify if there is a New Span available*/
            {
                newStartX = spans[newI*colsSpans+k];
                newEndX = spans[newI*colsSpans+k+1];
                if (startX <= newEndX && endX >= newStartX) /*Merge components*/
                {
                    label = components[i*(colsSpans/2)+(j/2)];          /*choose the startSpan label*/
                    for (int p=0; p<=i+1; p++)                          /*relabel*/
                    {
                        for(int q=0; q<colsSpans/2; q++)
                        {
                            if(components[p*(colsSpans/2)+q]==components[newI*(colsSpans/2)+(k/2)])
                            {
                                components[p*(colsSpans/2)+q] = label;
                            }
                        }
                    }
                }
            }
        }
    }

    double wall1 = getWallTime();
    double cpu1  = getCpuTime();
    cout << "Time Performance: ACCL serial" << endl;
    cout << "\tWall Time = " << (wall1 - wall0)*1000 << " ms" << endl;
    cout << "\tCPU Time  = " << (cpu1  - cpu0)*1000  << " ms" << endl;

    /*
     * Convert to a labeled image matrix
     */
    rgb *colors = new rgb[width*height];

    for (int index = 0; index < width*height; index++)
        colors[index] = randomRgb();

    for(int i=0; i<rowsSpans; i++)
    {
        for(int j=0; j<colsSpans ; j=j+2)
        {
            startX = spans[i*colsSpans+j];
            if(startX>=0)
            {
                endX = spans[i*colsSpans+j+1];
                for(int k=startX; k <=endX; k++)
                {
                    imRef(output, k, i)= colors[components[i*(colsSpans/2)+(j/2)]];
                }
            }
        }
    }
    //savePGM(output, "../../Data/out1.pgm");
    delete [] colors;
}
/*
 * pgmRead: read a pgm image file
 * Parameters:
 * - file:  ifstream
 *          path of the pgm image file
 * - buf:   char*
 *          buffer where information will be allocated
 */
void pgmRead(ifstream &file, char *buf)
{
    char doc[BUF_SIZE];
    char c;

    file >> c;
    while (c == '#')
    {
        file.getline(doc, BUF_SIZE);
        file >> c;
    }
    file.putback(c);

    file.width(BUF_SIZE);
    file >> buf;
    file.ignore();
}
/*
 * loadPGM: load pgm file and return it in a image<uchar> structure
 * Parameters:
 * - name:  const char*
 *          path of the pgm image file
 * Return:
 * - image<uchar>: image loaded in an uchar structure
 */
image<uchar> *loadPGM(const char *name)
{
    char buf[BUF_SIZE];

    /*
     * read header
     */
    std::ifstream file(name, std::ios::in | std::ios::binary);
    pgmRead(file, buf);
    if (strncmp(buf, "P5", 2))
        throw errorHandler();

    pgmRead(file, buf);
    int width = atoi(buf);
    pgmRead(file, buf);
    int height = atoi(buf);

    pgmRead(file, buf);
    if (atoi(buf) > UCHAR_MAX)
        throw errorHandler();

    /* read data */
    image<uchar> *im = new image<uchar>(width, height);
    file.read((char *)imPtr(im, 0, 0), width * height * sizeof(uchar));
    return im;
}
void savePGM(image<rgb> *im, const char *name)
{
    int width = im->width();
    int height = im->height();
    std::ofstream file(name, std::ios::out | std::ios::binary);

    file << "P6\n" << width << " " << height << "\n" << UCHAR_MAX << "\n";
    file.write((char *)imPtr(im, 0, 0), width * height * sizeof(rgb));
}

/*
 * imageUcharToInt: convert image from uchar to integer
 * Parameters:
 * - input: image<uchar>
 *          image in uchar to convert to integer values
 * Return:
 * - image<int>: image with integer values
 */
image<int> *imageUcharToInt(image<uchar> *input)
{
    int width = input->width();
    int height = input->height();
    image<int> *output = new image<int>(width, height, false);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            imRef(output, x, y) = imRef(input, x, y);
        }
    }
    return output;
}
int main()
{
    cout<<"Accelerated Connected Component Labeling" << endl;
    cout<<"========================================" << endl;
    cout<<"Loading input image..." << endl;
    // Number of frames
    // We need to define the input of nFrames
    image<uchar> *input = loadPGM("../data/8Frames.pgm");
    const uint nFrames = 8;

    const int width = input->width();
    const int height = input->height();

    /*
     * Declaration of Variables
     */
    image<int> *imInt = new image<int>(width, height);
    image<rgb> *output1 = new image<rgb>(width, height);
    image<rgb> *output2 = new image<rgb>(width, height);
    imInt = imageUcharToInt(input);

    const int rows = nFrames*512;
    const int cols = 512;
    const int imageSize = rows*cols;
    int *image = new int[imageSize];
    memcpy(image, imInt->data, rows * cols * sizeof(int));

    /*
     * Buffers
     */
    printf("Image reading was successful\n");
    const int colsSpans = ((cols+1)/2)*2; /*ceil(cols/2)*2*/
    const int spansSize = colsSpans*rows;
    const int componentsSize = (colsSpans/2)*rows;
    int *spans= new int[spansSize];
    int *components = new int[componentsSize];

    /*
     * Initialize
     */
    memset(spans, -1, spansSize*sizeof(int));
    memset(components, -1, componentsSize*sizeof(int));
    /*
     * CUDA
     */
    acclCuda(spans, components, image, nFrames, rows, cols);

    /*
     * Print output image
     */
    rgb *colors = new rgb[width*height];
    int startX, endX;
    for (int index = 0; index < rows*cols; index++)
        colors[index] = randomRgb();

    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<colsSpans; j=j+2)
        {
            startX = spans[i*colsSpans+j];
            if(startX>=0)
            {
                endX = spans[i*colsSpans+j+1];
                for(int k=startX; k <=endX; k++)
                {
                        if (components[i*(colsSpans/2)+(j/2)] != -1)
                        {
                            imRef(output2, k, i)= colors[components[i*(colsSpans/2)+(j/2)]];
                        }
                        else
                            printf("Error some spans weren't labeled\n");
                }
            }
        }
    }

    /*
     * Free memory
     */
    delete [] colors;
    savePGM(output2, "../data/out2.pgm");

    /*---------------- SERIAL --------------------*/
    int *spansSerial= new int[spansSize];
    acclSerial(image, spansSerial, components, rows, cols, output1);
    printf("Segmentation ended.\n");
    return 0;
}

