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
TARGET2 = intensities

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(TARGET1).c
	$(Ccc) $(TARGET1).c -o $(TARGET1)

$(TARGET2): $(TARGET2).c
	$(Ccc) $(TARGET2).c -o $(TARGET2)

clean:
	$(RM) $(TARGET1) $(TARGET2) *.o *.tar* *.core* *.pgm
	
run:
	./label
	./intensities
