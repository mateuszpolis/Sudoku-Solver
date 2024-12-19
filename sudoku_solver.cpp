#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <string>

#define SIZE 9
#define EMPTY 0
#define MAX_SOLUTIONS 1000000

// Function to check if placing 'num' at (row, col) is valid
bool is_valid(const std::vector<int>& board, int row, int col, int num) {
    // Check row
    for (int i = 0; i < SIZE; ++i) {
        if (board[row * SIZE + i] == num) {
            return false;
        }
    }

    // Check column
    for (int i = 0; i < SIZE; ++i) {
        if (board[i * SIZE + col] == num) {
            return false;
        }
    }

    // Check 3x3 grid
    int startRow = (row / 3) * 3;
    int startCol = (col / 3) * 3;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (board[(startRow + i) * SIZE + (startCol + j)] == num) {
                return false;
            }
        }
    }

    return true;
}

// Recursive backtracking function to solve Sudoku
void solve_sudoku(std::vector<int>& board, int cell, std::vector<std::vector<int>>& solutions, int& solution_count) {
    if (solution_count >= MAX_SOLUTIONS) {
        return; // Prevent excessive computation
    }

    // Find the next empty cell
    while (cell < SIZE * SIZE && board[cell] != EMPTY) {
        cell++;
    }

    // If no empty cell is left, a solution is found
    if (cell == SIZE * SIZE) {
        solutions.emplace_back(board);
        solution_count++;
        return;
    }

    int row = cell / SIZE;
    int col = cell % SIZE;

    // Try placing numbers 1 through 9 in the empty cell
    for (int num = 1; num <= SIZE; ++num) {
        if (is_valid(board, row, col, num)) {
            board[cell] = num; // Place the number

            // Recurse to solve the rest of the board
            solve_sudoku(board, cell + 1, solutions, solution_count);

            // Backtrack
            board[cell] = EMPTY;
        }
    }
}

// Structure to hold the results for one board
struct BoardResult {
    std::vector<std::vector<int>> solutions;
    int solution_count = 0;
};

// Thread function to solve a single board
void solve_board_thread(std::vector<int> board, BoardResult &result) {
    solve_sudoku(board, 0, result.solutions, result.solution_count);
}

int main(int argc, char* argv[]) {
    bool use_threads = false;
    bool save_output = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" || arg == "-t") {
            use_threads = true;
        } else if (arg == "--save-output" || arg == "-s") {
            save_output = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Usage: " << argv[0] << " [--threads | -t] [--save-output | -s]\n";
            return 1;
        }
    }

    // Read multiple boards from file
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

    // Read boards in the specified format (9 lines of 9 chars each)
    std::vector<std::vector<int>> boards(num_boards, std::vector<int>(SIZE * SIZE));
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
                boards[b][i * SIZE + j] = val;
            }
        }
    }
    infile.close();

    if (use_threads) {
        std::cout << "Solving " << num_boards << " Sudoku boards with parallel threads...\n";
    } else {
        std::cout << "Solving " << num_boards << " Sudoku boards sequentially (no threads)...\n";
    }

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<BoardResult> results(num_boards);

    if (use_threads) {
        // Parallel execution: one thread per board
        std::vector<std::thread> threads(num_boards);
        for (int b = 0; b < num_boards; ++b) {
            threads[b] = std::thread(solve_board_thread, boards[b], std::ref(results[b]));
        }

        // Wait for all threads to complete
        for (int b = 0; b < num_boards; ++b) {
            threads[b].join();
        }
    } else {
        // Sequential execution: solve boards one by one on the main thread
        for (int b = 0; b < num_boards; ++b) {
            solve_sudoku(boards[b], 0, results[b].solutions, results[b].solution_count);
        }
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Print the time taken to solve all boards
    std::cout << "Time taken to solve all Sudoku boards: " << duration.count() << " seconds\n";

    // Combine all solutions
    int total_solutions = 0;
    for (auto &res : results) {
        total_solutions += res.solution_count;
    }

    if (save_output) {
        // Open the output file
        std::ofstream outfile("solutions.txt");
        if (!outfile.is_open()) {
            std::cerr << "Failed to open solutions.txt for writing.\n";
            return 1;
        }

        // Write the total number of solutions
        outfile << "Total Solutions Found: " << total_solutions << "\n\n";

        int solution_index = 1;
        for (int b = 0; b < num_boards; ++b) {
            outfile << "Board " << b + 1 << " Solutions:\n";
            for (auto &sol : results[b].solutions) {
                outfile << "Solution " << solution_index++ << ":\n";
                for (int i = 0; i < SIZE; ++i) {
                    for (int j = 0; j < SIZE; ++j) {
                        outfile << std::setw(2) << sol[i * SIZE + j] << " ";
                    }
                    outfile << "\n";
                }
                outfile << "\n";
            }
            outfile << "\n";
        }

        // Close the file
        outfile.close();
        std::cout << "Solutions have been saved to solutions.txt\n";
    }

    return 0;
}