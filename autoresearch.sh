#!/bin/bash
set -euo pipefail

# Run the Python training script with a reduced number of epochs for faster iterations
# Standard run is 40 epochs; we use 10 as a proxy for optimization loops.
OUTPUT=$(python MLModel/AIModel/run/main_audio_jepa.py --epochs 10 2>&1)

# Extract the accuracy from the output
# We are looking for the final "BEST ACCURACY: XX.XX%" string
ACCURACY=$(echo "$OUTPUT" | grep "BEST ACCURACY:" | grep -o -E '[0-9]+\.[0-9]+' | head -n 1)

if [ -z "$ACCURACY" ]; then
    echo "Error: Could not extract accuracy from output. Script may have crashed."
    echo "Last 20 lines of output:"
    echo "$OUTPUT" | tail -n 20
    exit 1
fi

# Output the required metric format for pi-autoresearch
echo "METRIC accuracy=$ACCURACY"
