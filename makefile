# Compiler settings
CXX = g++
NVCC = nvcc

CXXFLAGS = -O2 -std=c++11 -pthread
NVCCFLAGS = -O2

# Targets
CPU_TARGET = sudoku_solver_cpu
GPU_TARGET = sudoku_solver_gpu

# Source files
CPU_SRC = sudoku_solver.cpp
GPU_SRC = sudoku_solver.cu

all: $(CPU_TARGET) $(GPU_TARGET)

$(CPU_TARGET): $(CPU_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(GPU_TARGET): $(GPU_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(CPU_TARGET) $(GPU_TARGET)