#!/bin/bash

export QT_QPA_PLATFORM=offscreen

# List of dataset choices
<<<<<<< HEAD
datasets=("Fe") #("PI" "TH" "QUH" "HC" "HH" ) #("PU" "TH" "PI" "UH13" "HC" "HH" "QUH") #"HH" "HC" "LK" "PI" "UH13" "SA" "PU" "PC" "QUH"
window_sizes=(12 14 16 18 20) #(2 4 6 8 10 12 14 16 18 20)
=======
datasets=("HC" "HH" "PI" "TH" "QUH") #("PU" "TH" "PI" "UH13" "HC" "HH" "QUH") #"HH" "HC" "LK" "PI" "UH13" "SA" "PU" "PC" "QUH"
window_sizes=(12) #(2 4 6 8 10 12 14 16 18 20)
>>>>>>> 20cea71 (initial_code)
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
command = f'python baseline_mamba.py --dataset $dataset --window_size $window_size --test_ratio $test_ratio --validation_ratio $validation_ratio'
subprocess.run(command, shell=True)

# Clear GPU memory by resetting the default graph
reset_default_graph()
tf.keras.backend.clear_session()
"
    done
done

# export QT_QPA_PLATFORM=offscreen

# # List of dataset choices
# datasets=("SA") #"HH" "HC" "LK" "PI" "UH13" "SA" "PU" "PC" "QUH"
# window_sizes=(10)
# # Define the train and validation ratios
# train_ratio=0.50
# validation_ratio=0.50
# test_ratio=0.90

# # Loop through each window size
# for window_size in "${window_sizes[@]}"
# do
#     # Loop through each dataset
#     for dataset in "${datasets[@]}"
#     do
#         echo "Running script for dataset: $dataset with patch size: $window_size"
#         python -c "
# import subprocess
# import tensorflow as tf
# from tensorflow.python.framework.ops import reset_default_graph

# # Set the GPU device (e.g., GPU:0)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#         tf.config.experimental.set_memory_growth(gpus[0], True)  # Optional: Enable memory growth
#     except RuntimeError as e:
#         print(e)

# # Alternatively, to use CPU only, uncomment the next line
# # tf.config.experimental.set_visible_devices([], 'GPU')

# # Run the main script
# command = f'python baseline.py --dataset $dataset --window_size $window_size --test_ratio $test_ratio --validation_ratio $validation_ratio'
# subprocess.run(command, shell=True)

# # Clear GPU memory by resetting the default graph
# reset_default_graph()
# tf.keras.backend.clear_session()
# "
#     done
# done
