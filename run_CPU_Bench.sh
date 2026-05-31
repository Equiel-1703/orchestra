#!/bin/bash

# Check if the user provided the .exs file path
if [ $# -ne 2 ]; then
    echo "Usage: $0 <path_to_exs_file> <input_size>"
    exit 1
fi

EXS_FILE="$1"
INPUT_SIZE="$2"

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" "$INPUT_SIZE"
done