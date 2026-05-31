#!/bin/bash

EXS_FILE="benchmarks/cpu/dot_product_nx.exs"

# 10 million elements
INPUT_SIZE=10000000

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done

# 30 million elements
INPUT_SIZE=30000000

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done

# 50 million elements
INPUT_SIZE=50000000

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done