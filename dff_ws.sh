#!/bin/bash

export QT_QPA_PLATFORM=offscreen

# List of dataset choices
datasets=("HH" "HC" "PI" "TH" "QUH") #"HH" "HC" "PI" "TH" "QUH") #("PU" "TH" "PI" "UH13" "HC" "HH" "QUH") #"HH" "HC" "LK" "PI" "UH13" "SA" "PU" "PC" "QUH"
window_sizes=(12 8 10 14 16) #(8 10 12 14) 
# Define the train and validation ratios
train_ratio=0.50
validation_ratio=0.50
test_ratio=0.90

# Specify the GPU ID(s) to use (e.g., "0" for GPU:0, "1" for GPU:1, or "0,1" for multiple GPUs)
GPU_IDS="0"  # Change this to the desired GPU ID(s), e.g., "1" or "0,1"

# Loop through each window size
for window_size in "${window_sizes[@]}"
do
    # Loop through each dataset
    for dataset in "${datasets[@]}"
    do
        echo "Running script for dataset: $dataset with patch size: $window_size on GPU(s): $GPU_IDS"
        python -c "
import subprocess
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph

# Set the specific GPU(s) to use
gpu_ids = '$GPU_IDS'.split(',')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Filter the GPUs based on the specified IDs
        selected_gpus = [gpus[int(i)] for i in gpu_ids if int(i) < len(gpus)]
        tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')
        for gpu in selected_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Enable memory growth for each GPU
        print(f'Using GPU(s): {[gpu.name for gpu in selected_gpus]}')
    except RuntimeError as e:
        print(f'Error setting GPU(s): {e}')
else:
    print('No GPUs available, running on CPU')

# Run the main script
command = f'python main.py --dataset $dataset --window_size $window_size --test_ratio $test_ratio --validation_ratio $validation_ratio'
subprocess.run(command, shell=True)

# Clear GPU memory by resetting the default graph
reset_default_graph()
tf.keras.backend.clear_session()
"
    done
done

