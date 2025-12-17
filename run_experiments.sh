#!/bin/bash

PYTHON="/home/ldavilr/repos/SetBasedSTN/.setbasedstnenv/bin/python"

# List of all required script paths
SCRIPTS=(

    "/home/ldavilr/repos/SetBasedSTN/hexaly_benchmark.py"
    "/home/ldavilr/repos/SetBasedSTN/hexaly_benchmark_2.py"
    # "/home/ldavilr/repos/SetBasedSTN/hexaly_benchmark_3.py"
    "/home/ldavilr/repos/SetBasedSTN/hexaly_benchmark_3600.py"
    "/home/ldavilr/repos/SetBasedSTN/hexaly_benchmark_2_3600.py"
)

# Check if all files exist
ALL_EXIST=true
for SCRIPT in "${SCRIPTS[@]}"; do
    if [ ! -f "$SCRIPT" ]; then
        echo "Missing file: $SCRIPT"
        ALL_EXIST=false
    fi
done

# If all files exist, run them
if [ "$ALL_EXIST" = true ]; then

    for SCRIPT in "${SCRIPTS[@]}"; do
        echo "Running: $SCRIPT"
        "$PYTHON" "$SCRIPT"
    done

else
    echo "Aborting: One or more files are missing."
fi