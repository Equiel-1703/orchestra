#!/bin/bash

EXS_FILE="benchmarks/cpu/dot_product_elixir.exs"

# 200 million elements
INPUT_SIZE=200000000

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done

# 300 million elements
INPUT_SIZE=300000000

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done

# 400 million elements
INPUT_SIZE=400000000

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done