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
TARGET1 = label-in
TARGET2 = label-out
TARGET3 = intensities

all: $(TARGET1) $(TARGET2) $(TARGET3)

$(TARGET1): $(TARGET1).c
	$(Ccc) $(TARGET1).c -o $(TARGET1)

$(TARGET2): $(TARGET2).c
	$(Ccc) $(TARGET2).c -o $(TARGET2)

$(TARGET3): $(TARGET3).c
	$(Ccc) $(TARGET3).c -o $(TARGET3)

run:
	./label-in
	./label-out
	./intensities
	ln -s circle-1024.label-in.pgm circle.label.pgm
	ln -s circle-1024.intensities.pgm circle.intensities.pgm

clean:
	$(RM) $(TARGET1) $(TARGET2) $(TARGET3) *.o *.tar* *.core* *.pgm
