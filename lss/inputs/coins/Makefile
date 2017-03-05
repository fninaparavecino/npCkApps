# Compiler
CXX = g++
Ccc = gcc

# Remove function
RM = rm -f
 
# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
CXXFLAGS = 
CccFLAGS = -lm

debug: CXXFLAGS += -DDEBUG -g -Wall
debug: CccFLAGS += -DDEBUG
debug: all

# The build target executable:
TARGET1 = label

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(TARGET1).c
	$(Ccc) $(TARGET1).c -o $(TARGET1)

clean:
	$(RM) $(TARGET1) *.o *.tar* *.core* 
	
run:
	./label
	./intensities
