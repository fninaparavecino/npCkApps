# enviroment
CUDA_VERSION = cuda-7.5
CPP = g++

NVCC = /usr/local/$(CUDA_VERSION)/bin/nvcc
SASSI = /usr/local/sassi7/bin/nvcc
SASSI_LIB_PATH = /home/fanny/Desktop/Research/Development/SASSI/instlibs/lib
CUDA_LIB_PATH = /usr/local/$(CUDA_VERSION)/lib64

SOURCEDIR = .

CPP_SOURCES = $(SOURCEDIR)/accl_main.cpp
CU_SOURCES = $(SOURCEDIR)/accl.cu

CFLAGS =

NVCCFLAGS = -g -O3

# Define ptxas injections
BEFORE_MEM = -Xptxas --sassi-inst-before="memory"

# Define args to pass to handlers
BEFORE_MEM_INFO = -Xptxas --sassi-before-args="mem-info"

# We rely heavily on CUPTI to let us know when kernels are launched.
CUPTI_LIB_DIR = /usr/local/sassi7/extras/CUPTI/lib64
CUPTI = $(CUPTI_LIB_DIR)
export LD_LIBRARY_PATH := $(INST_LIB_DIR):$(CUPTI_LIB_DIR):$(LD_LIBRARY_PATH)

ARCH = -gencode arch=compute_35,code=sm_35

EXE = $(SOURCEDIR)/accl

DEVICE_OBJS = $(SOURCEDIR)/accl_cu.o

LINK_OBJS = $(SOURCEDIR)/accl_cu_dlink.o

all: acclNP

acclNP: acclNP.cu
	$(NVCC) $(NVCCFLAGS) $(ARCH) $< -o acclNP -L$(CUDA_LIB_PATH) -rdc=true -lcudadevrt

accl_sassi_none: acclNP.cu
	$(SASSI) -dc $(NVCCFLAGS) $(ARCH) $< -o accl_sassi.o -L$(CUDA_LIB_PATH) -lcudadevrt
	$(SASSI) accl_sassi.o -o accl_sassi $(ARCH) -L$(SASSI_LIB_PATH) -L$(CUPTI) -lcupti -lcudadevrt

accl_sassi: acclNP.cu
	$(SASSI) -c $(ARCH) $(BEFORE_MEM) $(BEFORE_MEM_INFO) -dc $(NVCCFLAGS) -o accl_sassi.o $<
	$(SASSI) accl_sassi.o -o accl_sassi $(ARCH) -L$(SASSI_LIB_PATH) -lmemdiverge -L$(CUPTI) -lcupti -lcudadevrt

clean:
	rm -f $(DEVICE_OBJS)
	rm -f $(LINK_OBJS)
	rm -f $(EXE)
	rm -f accl_sassi
	rm -f acclNP
