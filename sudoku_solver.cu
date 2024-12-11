#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#define SIZE 9
#define NUM_THREADS 256
#define MAX_BOARDS 10000000 // Adjust as needed
#define MAX_ITERATIONS 1000 // Safety limit to prevent infinite loops

// Device function to check if placing num at (row, col) is valid
__device__ bool is_valid(int* board, int row, int col, int num) {
    // Check row
    for (int i = 0; i < SIZE; ++i)
        if (board[row * SIZE + i] == num)
            return false;

    // Check column
    for (int i = 0; i < SIZE; ++i)
        if (board[i * SIZE + col] == num)
            return false;

    // Check 3x3 grid
    int startRow = (row / 3) * 3;
    int startCol = (col / 3) * 3;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (board[(startRow + i) * SIZE + startCol + j] == num)
                return false;

    return true;
}

// Modify the kernel as follows:

__global__ void solve_sudoku_kernel(
    int* current_boards, 
    int* next_boards, 
    int* next_count, 
    int* solutions, 
    int* solution_count, 
    int num_current_boards
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_current_boards) return;

    int board[SIZE * SIZE];
    for (int i = 0; i < SIZE * SIZE; ++i)
        board[i] = current_boards[idx * SIZE * SIZE + i];

    // Find the first empty cell
    int empty_pos = -1;
    for (int i = 0; i < SIZE * SIZE; ++i) {
        if (board[i] == 0) {
            empty_pos = i;
            break;
        }
    }

    // If no empty cell, record solution
    if (empty_pos == -1) {
        int sol_idx = atomicAdd(solution_count, 1);
        if (sol_idx < MAX_BOARDS) { // Check we don't exceed our solution buffer
            for (int i = 0; i < SIZE * SIZE; ++i)
                solutions[sol_idx * SIZE * SIZE + i] = board[i];
        } else {
            // If we exceed the solution buffer, we might stop or just skip writing
            // Consider reverting the atomicAdd if desired, but that's tricky
            // For simplicity, we will just skip writing
        }
        return;
    }

    int row = empty_pos / SIZE;
    int col = empty_pos % SIZE;
    bool any_valid = false;

    for (int num = 1; num <= SIZE; ++num) {
        if (is_valid(board, row, col, num)) {
            any_valid = true;
            int new_board_idx = atomicAdd(next_count, 1);
            if (new_board_idx < MAX_BOARDS) {
                // Write the new board
                for (int i = 0; i < SIZE * SIZE; ++i)
                    next_boards[new_board_idx * SIZE * SIZE + i] = board[i];
                next_boards[new_board_idx * SIZE * SIZE + empty_pos] = num;
            } else {
                // If we exceed MAX_BOARDS for expansions, we should consider stopping further expansions
                // One strategy is to just skip creating more boards
                // Another is to reset next_count back by one atomic operation, but this can be complex
                // For now, we just skip writing more boards
            }
        }
    }
}

int main() {
    // Read boards from a file
    std::string filename = "boards.txt";
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open " << filename << " for reading.\n";
        return 1;
    }

    int num_boards;
    infile >> num_boards;
    if (num_boards <= 0) {
        std::cerr << "No boards to solve.\n";
        return 1;
    }

    // Read all boards into a host vector
    // We'll store all boards consecutively: board0(81 ints), board1(81 ints), ...
    std::vector<int> host_boards(num_boards * SIZE * SIZE);
    for (int b = 0; b < num_boards; ++b) {
        for (int i = 0; i < SIZE; ++i) {
            std::string line;
            infile >> line;
            if (line.size() != SIZE) {
                std::cerr << "Invalid board line encountered.\n";
                return 1;
            }
            for (int j = 0; j < SIZE; ++j) {
                char c = line[j];
                if (c < '0' || c > '9') {
                    std::cerr << "Invalid character in board: " << c << std::endl;
                    return 1;
                }
                int val = c - '0';
                host_boards[b * SIZE * SIZE + i * SIZE + j] = val;
            }
        }
    }

    infile.close();

    // Allocate memory for current boards and next boards on the GPU
    int* d_current_boards;
    int* d_next_boards;
    cudaMalloc(&d_current_boards, sizeof(int) * SIZE * SIZE * MAX_BOARDS);
    cudaMalloc(&d_next_boards, sizeof(int) * SIZE * SIZE * MAX_BOARDS);

    // Allocate memory for solutions on the GPU
    int* d_solutions;
    cudaMalloc(&d_solutions, sizeof(int) * SIZE * SIZE * MAX_BOARDS); // Adjust size as needed

    // Allocate memory for counts
    int* d_next_count;
    int* d_solution_count;
    cudaMalloc(&d_next_count, sizeof(int));
    cudaMalloc(&d_solution_count, sizeof(int));

    // Initialize next_count and solution_count to zero
    cudaMemset(d_next_count, 0, sizeof(int));
    cudaMemset(d_solution_count, 0, sizeof(int));

    // Copy all boards to current_boards on device
    cudaMemcpy(d_current_boards, host_boards.data(), sizeof(int) * SIZE * SIZE * num_boards, cudaMemcpyHostToDevice);

    // Initialize the number of current boards to the number of boards we read
    int num_current_boards = num_boards;

    // Define kernel launch parameters
    int threads_per_block = NUM_THREADS;

    int iteration = 0; // Iteration counter

    auto start_algorithm = std::chrono::high_resolution_clock::now();

    // Iterate until all boards are processed or maximum iterations reached
    while (num_current_boards > 0 && iteration < MAX_ITERATIONS) {
        iteration++;

        // Reset next_count to zero
        cudaMemset(d_next_count, 0, sizeof(int));

        // Launch kernel to process current boards
        solve_sudoku_kernel<<<(num_current_boards + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
            d_current_boards,
            d_next_boards,
            d_next_count,
            d_solutions,
            d_solution_count,
            num_current_boards
        );

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed at iteration " << iteration << ": " 
                      << cudaGetErrorString(err) << std::endl;
            break;
        }

        // Synchronize to ensure kernel completion
        cudaDeviceSynchronize();

        // Get the number of next boards
        int h_next_count;
        cudaMemcpy(&h_next_count, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);

        // Get the number of solutions found so far
        int h_solution_count;
        cudaMemcpy(&h_solution_count, d_solution_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_next_count >= MAX_BOARDS) {
            std::cerr << "MAX_BOARDS limit reached. Stopping expansions." << std::endl;            
            break;
        }

        // Swap current_boards and next_boards
        int* temp = d_current_boards;
        d_current_boards = d_next_boards;
        d_next_boards = temp;

        // Update the number of current boards for the next iteration
        num_current_boards = h_next_count;

        // If no new boards are generated, exit the loop
        if (h_next_count == 0) {
            break;
        }
    }

    auto end_algorithm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_algorithm = end_algorithm - start_algorithm;
    // std::cout << "Time taken for important algorithm stages: " << duration_algorithm.count() << " seconds\n";
    std::cout << duration_algorithm.count() << "\n";

    // Get the total number of solutions
    int total_solutions;
    cudaMemcpy(&total_solutions, d_solution_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Allocate host memory for solutions
    std::vector<int> solutions(total_solutions * SIZE * SIZE);
    if (total_solutions > 0) {
        cudaMemcpy(solutions.data(), d_solutions, sizeof(int) * SIZE * SIZE * total_solutions, cudaMemcpyDeviceToHost);
    }

    // Open the output file
    std::ofstream outfile("solutions.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open solutions.txt for writing.\n";
        // Free memory before exiting
        cudaFree(d_current_boards);
        cudaFree(d_next_boards);
        cudaFree(d_solutions);
        cudaFree(d_next_count);
        cudaFree(d_solution_count);
        return 1;
    }

    // Write the total number of solutions
    outfile << "Total Solutions Found: " << total_solutions << "\n\n";

    // Write each solution to the file
    for (int s = 0; s < total_solutions; ++s) {
        outfile << "Solution " << s + 1 << ":\n";
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                outfile << solutions[s * SIZE * SIZE + i * SIZE + j] << " ";
            }
            outfile << "\n";
        }
        outfile << "\n";
    }

    // Close the file
    outfile.close();
    // std::cout << "Solutions have been saved to solutions.txt\n";

    // Free memory
    cudaFree(d_current_boards);
    cudaFree(d_next_boards);
    cudaFree(d_solutions);
    cudaFree(d_next_count);
    cudaFree(d_solution_count);

    // Check if maximum iterations were reached
    if (iteration >= MAX_ITERATIONS) {
        std::cerr << "Reached maximum number of iterations (" << MAX_ITERATIONS << "). "
                  << "The program may be stuck in an infinite loop.\n";
    }

    return 0;
}