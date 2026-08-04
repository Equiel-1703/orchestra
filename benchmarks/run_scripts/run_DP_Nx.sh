#!/bin/bash

EXS_FILE="../cpu/dot_product_nx.exs"

# 10 million elements
INPUT_SIZE=10000000

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done

# 15 million elements
INPUT_SIZE=15000000

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done

# 20 million elements
INPUT_SIZE=200000000

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done