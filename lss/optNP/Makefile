# environment
SM := 35

GCC := g++
NVCC := nvcc

# Remove function
RM = rm -f
 
# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)

NVCCFLAGS := -O3
GccFLAGS = -fopenmp -O3 

#debug: GccFLAGS += -DDEBUG -g -Wall
debug: NVCCFLAGS += -g -G
debug: all

# The build target executable:
TARGET = lss

all: build

build: $(TARGET)

$(TARGET): src/dlink.o src/main.o src/alg/$(TARGET).o src/img/imghandler.o src/alg/locationhandler.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -link

src/dlink.o: src/alg/$(TARGET).o 
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

src/main.o: src/main.cpp src/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@

src/alg/locationhandler.o: src/alg/locationhandler.cpp src/alg/locationhandler.h src/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@
	
src/alg/$(TARGET).o: src/alg/$(TARGET).cu src/config.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 

src/img/imghandler.o: src/img/imghandler.cpp src/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@
	
clean:
	$(RM) $(TARGET) *.o src/*.o src/*/*.o *.tar* *.core*
