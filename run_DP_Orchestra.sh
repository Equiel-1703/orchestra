#!/bin/bash

EXS_FILE="benchmarks/cpu/dot_product.exs"

# 30 million elements
INPUT_SIZE=30000000

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done

# 10 million elements
INPUT_SIZE=10000000

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done