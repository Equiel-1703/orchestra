#!/bin/bash

# Check if the user provided the .exs file path
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_exs_file>"
    exit 1
fi

EXS_FILE="$1"

# Execute 30 times
for i in $(seq 1 30); do
    mix run "$EXS_FILE" 1024 | grep "Nx|Orchestra"
done