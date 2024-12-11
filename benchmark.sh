#!/bin/bash

# Initialize variables to store the cumulative time results
sum_seq=0
sum_seq_threads=0
sum_gpu=0

# Number of repetitions
repetitions=5

# Run each program 5 times and calculate the total time
for i in $(seq 1 $repetitions); do
    # Run the first program
    result_seq=$(./sudoku_solver_seq)
    sum_seq=$(echo "$sum_seq + $result_seq" | bc)

    # Run the second program
    result_seq_threads=$(./sudoku_solver_seq -t)
    sum_seq_threads=$(echo "$sum_seq_threads + $result_seq_threads" | bc)

    # Run the third program
    result_gpu=$(./sudoku_solver)
    sum_gpu=$(echo "$sum_gpu + $result_gpu" | bc)
done

# Calculate the averages
avg_seq=$(echo "scale=10; $sum_seq / $repetitions" | bc)
avg_seq_threads=$(echo "scale=10; $sum_seq_threads / $repetitions" | bc)
avg_gpu=$(echo "scale=10; $sum_gpu / $repetitions" | bc)

# Print the results in the specified format
echo "Suoku Solver on CPU (sequential): $avg_seq seconds"
echo "Suoku Solver on CPU (sequential with parallel threads): $avg_seq_threads seconds"
echo "Suoku Solver on GPU (parallel): $avg_gpu seconds"