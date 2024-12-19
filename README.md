# Sudoku Solver Project

This repository contains two Sudoku solver implementations—one leveraging GPU parallelization using CUDA, and the other running on the CPU using a traditional sequential backtracking approach. Additionally, it includes a script to generate sample Sudoku boards at various difficulty levels.

## Overview

## GPU-Based Sudoku Solver
- File: sudoku_solver.cu
- Approach: Utilizes parallel computation on the GPU.
- Methodology: Employs a breadth-first search (BFS)-like expansion of Sudoku states. Each kernel invocation attempts to fill empty cells with valid numbers, generating new board states in parallel.
- Use Case: Ideal for handling large batches of complex Sudoku puzzles or very difficult puzzles that benefit from massive parallelism.

## CPU-Based Sudoku Solver
- File: sudoku_solver.cpp
- Approach: Uses a traditional sequential backtracking algorithm.
- Methodology: Checks each empty cell and attempts valid placements of digits 1–9, backtracking as needed. While it can be time-consuming for complex puzzles, it is straightforward and does not require specialized GPU hardware.
- Optional Parallelism: By using the --threads or -t flag, you can solve multiple boards concurrently in separate CPU threads.

## Board Input Format

The programs read Sudoku boards from an input file named boards.txt by default. The file format is:
  1. The first line contains an integer N, the number of Sudoku boards.
	2. Followed by N Sudoku boards, each described by 9 lines of 9 characters (0 represents an empty cell, and 1–9 represent given digits).

```txt
2

000000000
000000000
000000000
000000000
000000000
000000000
000000000
000000000
000000000

530070000
600195000
098000060
800060003
400803001
700020006
060000280
000419005
000080079
```

This file describes 2 boards. The first is completely empty, and the second includes some given numbers.

## Generating Sample Boards

A generator.js script (using the sudoku-puzzle tool) is included to generate a large number of Sudoku boards with varying difficulties. To use it:
1.	Install Node.js.
2.	Run npm install sudoku-puzzle (if needed).
3.	Execute node generator.js to generate a batch of boards at a specified hardness level.
4.	The generated boards can be directed into boards.txt to serve as input for the solvers.

## Building the Programs

A Makefile is provided for convenient building:
- To build the CPU solver:

```bash
make sudoku_solver_cpu
```

- To build the GPU solver (requires NVIDIA CUDA and nvcc):

```bash
make sudoku_solver_gpu
``` 

- To build both:

```bash
make
```

- To clean up executables:

```bash 
make clean
```

## Usage

### CPU Solver

**Command**:

```bash
./sudoku_solver_cpu [--threads | -t] [--save-output | -s]
```

**Options**:
- --threads or -t: Use multiple threads (one thread per board).
-	--save-output or -s: Save the found solutions to solutions.txt.

**Example**:

```bash
./sudoku_solver_cpu --threads --save-output
```

This will solve all boards in boards.txt using multiple threads and save the solutions to solutions.txt.

### GPU Solver

**Command**:

```bash
./sudoku_solver_gpu [--save-output | -s]
```

**Options**:

- --save-output or -s: Save the found solutions to solutions.txt.

**Example**:

```bash
./sudoku_solver_gpu --save-output
```

This will solve all boards in boards.txt using the GPU and save the solutions to solutions.txt.

## Output

- Console Output:
Both programs print the total time taken to solve all Sudoku boards, and the total number of solutions found.
- File Output:
If the --save-output or -s flag is used, all solutions will be written to solutions.txt. Each solution is printed in a human-readable 9x9 grid format.

## Performance Considerations
- For a small number of simple boards, the CPU solver may be sufficient.
- For large-scale experiments (e.g., thousands or tens of thousands of boards) or extremely difficult puzzles, the GPU solver can provide dramatic speedups due to its parallel execution model.
- The CPU solver supports multithreading, which can speed up solving multiple boards on systems with multiple CPU cores.

## Further Reading

Refer to the included PDF report (Sudoku Solver Project Report) for a detailed comparative analysis of CPU-based and GPU-based Sudoku solving, including benchmarks and discussions on when each approach is most beneficial.