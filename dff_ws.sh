#!/usr/bin/env bash

set -euo pipefail

# Useful on a headless server.
export QT_QPA_PLATFORM=offscreen

# Keep CUDA device numbering consistent with nvidia-smi.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Expose only physical GPU 1 to TensorFlow.
export CUDA_VISIBLE_DEVICES=1

# Dataset identifiers accepted by baseline.py.
# TH = Tangdaowan
# PI = Pingan
datasets=("HH" "HC" "TH" "PI")

# Your patch extraction currently expects even window sizes.
patch_sizes=(12 14)

# Experiment settings.
number_of_bands=15
train_ratio=0.25
validation_ratio=0.25
test_ratio=0.50
epochs=50
batch_size=56

echo "======================================================"
echo "HSI Hybrid-ViT batch experiments"
echo "Physical GPU selected: ${CUDA_VISIBLE_DEVICES}"
echo "Datasets: ${datasets[*]}"
echo "Patch sizes: ${patch_sizes[*]}"
echo "======================================================"

# Confirm that physical GPU 1 exists.
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

if (( gpu_count <= 1 )); then
    echo "ERROR: GPU 1 is unavailable."
    echo "Detected GPU count: ${gpu_count}"
    nvidia-smi
    exit 1
fi

echo "GPU status before starting:"
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv

for dataset in "${datasets[@]}"; do
    for ws in "${patch_sizes[@]}"; do
        echo
        echo "------------------------------------------------------"
        echo "Dataset: ${dataset}"
        echo "Window size: ${ws}"
        echo "Physical GPU: ${CUDA_VISIBLE_DEVICES}"
        echo "------------------------------------------------------"

        python baseline_with_github_hybrid_vit.py \
            --dataset "${dataset}" \
            --window_size "${ws}" \
            --number_of_bands "${number_of_bands}" \
            --train_ratio "${train_ratio}" \
            --validation_ratio "${validation_ratio}" \
            --test_ratio "${test_ratio}" \
            --epochs "${epochs}" \
            --batch_size "${batch_size}"

        # The training process has exited, so its CUDA memory should be released.
        sleep 3
    done
done

echo
echo "All  baseline experiments completed."






# #!/usr/bin/env bash

# set -euo pipefail

# # Required for servers without a display.
# export QT_QPA_PLATFORM=offscreen

# # Make CUDA numbering correspond to PCI/nvidia-smi ordering.
# export CUDA_DEVICE_ORDER=PCI_BUS_ID

# # Physical GPU 1 only.
# export CUDA_VISIBLE_DEVICES=1

# datasets=("HC" "HH" "PI" "TH" "QUH")
# window_sizes=(12)

# train_ratio=0.50
# validation_ratio=0.50
# test_ratio=0.90

# epochs=50
# batch_size=56
# number_of_bands=15

# echo "============================================================"
# echo "Experiment launcher"
# echo "Physical CUDA GPU selected: ${CUDA_VISIBLE_DEVICES}"
# echo "Datasets: ${datasets[*]}"
# echo "Window sizes: ${window_sizes[*]}"
# echo "============================================================"

# # Confirm that physical GPU 1 exists.
# if ! nvidia-smi --query-gpu=index,name,memory.total,memory.used \
#     --format=csv,noheader | grep -q "^1,"; then
#     echo "ERROR: Physical GPU 1 was not found."
#     nvidia-smi
#     exit 1
# fi

# echo "Current GPU status:"
# nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
#     --format=csv

# for window_size in "${window_sizes[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         echo
#         echo "============================================================"
#         echo "Dataset: ${dataset}"
#         echo "Patch size: ${window_size}"
#         echo "Physical GPU: ${CUDA_VISIBLE_DEVICES}"
#         echo "============================================================"

#         python main.py \
#             --dataset "${dataset}" \
#             --window_size "${window_size}" \
#             --number_of_bands "${number_of_bands}" \
#             --train_ratio "${train_ratio}" \
#             --validation_ratio "${validation_ratio}" \
#             --test_ratio "${test_ratio}" \
#             --epochs "${epochs}" \
#             --batch_size "${batch_size}"
#     done
# done

# echo
# echo "All experiments completed."

# !/bin/bash

# export QT_QPA_PLATFORM=offscreen

# # List of dataset choices
# datasets=( "HC" "HH" "PI" "TH" "QUH")  #("PU" "TH" "PI" "UH13" "HC" "HH" "QUH") 
# window_sizes=(12) #(8 10 12 14) 
# # Define the train and validation ratios
# train_ratio=0.25
# validation_ratio=0.25
# test_ratio=0.50

# # Specify the GPU ID(s) to use (e.g., "0" for GPU:0, "1" for GPU:1, or "0,1" for multiple GPUs)
# GPU_IDS="0"  # Change this to the desired GPU ID(s), e.g., "1" or "0,1"

# # Loop through each window size
# for window_size in "${window_sizes[@]}"
# do
#     # Loop through each dataset
#     for dataset in "${datasets[@]}"
#     do
#         echo "Running script for dataset: $dataset with patch size: $window_size on GPU(s): $GPU_IDS"
#         python -c "
# import subprocess
# import tensorflow as tf
# from tensorflow.python.framework.ops import reset_default_graph

# # Set the specific GPU(s) to use
# gpu_ids = '$GPU_IDS'.split(',')
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Filter the GPUs based on the specified IDs
#         selected_gpus = [gpus[int(i)] for i in gpu_ids if int(i) < len(gpus)]
#         tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')
#         for gpu in selected_gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)  # Enable memory growth for each GPU
#         print(f'Using GPU(s): {[gpu.name for gpu in selected_gpus]}')
#     except RuntimeError as e:
#         print(f'Error setting GPU(s): {e}')
# else:
#     print('No GPUs available, running on CPU')

# # Run the main script
# command = f'python main.py --dataset $dataset --window_size $window_size --test_ratio $test_ratio --validation_ratio $validation_ratio'
# subprocess.run(command, shell=True)

# # Clear GPU memory by resetting the default graph
# reset_default_graph()
# tf.keras.backend.clear_session()
# "
#     done
# done

