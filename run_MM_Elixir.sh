#!/bin/bash

EXS_FILE="benchmarks/cpu/mm_elixir.exs"

# 256 x 256
INPUT_SIZE=256

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done

# 512 x 512
INPUT_SIZE=512

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done

# 1024 x 1024
INPUT_SIZE=1024

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done