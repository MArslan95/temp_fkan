#!/bin/bash

# Set environment variable for headless matplotlib plotting
export QT_QPA_PLATFORM=offscreen

# List of dataset choices
datasets=("Fe")
window_sizes=(8)

# Define train, validation, and test ratios (should sum to ~1 or match Python logic)
train_ratio=0.50
validation_ratio=0.50
test_ratio=0.90

# Specify the GPU ID(s) to use (e.g., "0" for GPU:0, "0,1" for multiple GPUs)
GPU_IDS="0,1"

# Define other parameters required by the Python script

# Base output directory for logs
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Loop through each window size
for window_size in "${window_sizes[@]}"
do
    # Loop through each dataset
    for dataset in "${datasets[@]}"
    do
        echo "Running script for dataset: $dataset with window size: $window_size on GPU(s): $GPU_IDS"
        
        # Create a unique log file for this run
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="${LOG_DIR}/${dataset}_ws${window_size}_${TIMESTAMP}.log"
        
        # Run the Python script with CUDA_VISIBLE_DEVICES to control GPU usage
        CUDA_VISIBLE_DEVICES="$GPU_IDS" python dis_main.py \
            --dataset "$dataset" \
            --window_size "$window_size" \
            --dl_method "$dl_method" \
            --train_ratio "$train_ratio" \
            --validation_ratio "$validation_ratio" \
            --test_ratio "$test_ratio" \
        # Capture the process ID
        PID=$!
        
        # Wait for the process to complete and check its exit status
        wait $PID
        if [ $? -ne 0 ]; then
            echo "Error: Run failed for dataset $dataset with window size $window_size. Check $LOG_FILE for details."
            continue
        fi
        
        echo "Completed run for dataset: $dataset with window size: $window_size. Output logged to $LOG_FILE"
        
        # Clear TensorFlow session to release GPU memory
        python -c "
import tensorflow as tf
tf.keras.backend.clear_session()
" 2>/dev/null
    done
done

echo "All runs completed. Logs are stored in $LOG_DIR"