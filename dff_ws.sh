#!/usr/bin/env bash
# Reviewer 2: GitHub-style connected class regions + STRICT zero patch leakage.
# Dataset HC and HH; independent 12x12 source-pixel footprints across train/val/test.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1
PYTHON_SCRIPT="${PYTHON_SCRIPT:-main.py}"
GPU_IDS="${GPU_IDS:-1}"
datasets=(HC HH PI)
window_sizes=(12)
seeds=(123 345 2024) #(42) #(123 345 2024)
dl_method=PCA
number_of_bands=15
batch_size=56
EPOCHS="${EPOCHS:-50}"
SEARCH_TRIALS="${SEARCH_TRIALS:-512}"
SPATIAL_BUFFER="${SPATIAL_BUFFER:-0}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-}"
VAL_PER_CLASS="${VAL_PER_CLASS:-}"
MIN_TEST_PER_CLASS="${MIN_TEST_PER_CLASS:-1}"
# Verify on the actual data before training; use RUN_MODE=train afterwards.
RUN_MODE="${RUN_MODE:-train}"
case "$RUN_MODE" in
  train) flags=(--deterministic --prediction_map "${PREDICTION_MAP:-test}") ;;
  split_only) flags=(--verify_split_only) ;;
  *) echo 'RUN_MODE must be train or split_only' >&2; exit 2 ;;
esac
[[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing $PYTHON_SCRIPT" >&2; exit 2; }
[[ -d ./hsi_datasets ]] || { echo 'Missing ./hsi_datasets' >&2; exit 2; }
if [[ -n "$TRAIN_PER_CLASS" ]]; then flags+=(--connected_train_pixels "$TRAIN_PER_CLASS"); fi
if [[ -n "$VAL_PER_CLASS" ]]; then flags+=(--connected_validation_pixels "$VAL_PER_CLASS"); fi
flags+=(--min_test_per_class "$MIN_TEST_PER_CLASS")
mkdir -p logs
for dataset in "${datasets[@]}"; do
  for window_size in "${window_sizes[@]}"; do
    for seed in "${seeds[@]}"; do
      log="logs/${dataset}_github_connected_ws${window_size}_seed${seed}_$(date +%Y%m%d_%H%M%S).log"
      echo "Protocol=connected_patchsafe Dataset=$dataset WS=$window_size split/model seed=$seed GPU=$GPU_IDS mode=$RUN_MODE"
    #   echo "GitHub quota=min original GT class count // 10 for train/class; val=train quota // 5. 5/5/90 NOT used."
      if PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
        python -u "$PYTHON_SCRIPT" \
          --dataset "$dataset" --window_size "$window_size" \
          --dl_method "$dl_method" --number_of_bands "$number_of_bands" \
          --batch_size "$batch_size" --epochs "$EPOCHS" \
          --seed "$seed" --split_seed "$seed" \
          --split_protocol connected --search_trials "$SEARCH_TRIALS" \
          --spatial_buffer "$SPATIAL_BUFFER" \
          "${flags[@]}" 2>&1 | tee "$log"; then
        echo "SUCCESS: $dataset seed=$seed"
      else
        rc=$?
        echo "FAILED (exit $rc): $dataset seed=$seed; see $log" >&2
        exit "$rc"
      fi
    done
  done
done
echo 'Completed all requested HC/HH connected-region experiments.'












#!/usr/bin/env bash
# Spatially disjoint FKAN experiment launcher (for the seeded main file).
# set -Eeuo pipefail

# # main.py loads ./hsi_datasets relative to the current working directory.
# cd "$(dirname "${BASH_SOURCE[0]}")"

# export QT_QPA_PLATFORM=offscreen
# export MPLBACKEND=Agg
# export TF_FORCE_GPU_ALLOW_GROWTH=true
# export PYTHONUNBUFFERED=1

# PYTHON_SCRIPT="${PYTHON_SCRIPT:-main.py}"
# GPU_IDS="${GPU_IDS:-1}"                 # One GPU; two visible GPUs != distributed training.
# datasets=("HC" "HH")                        # HC is supported by the supplied dataset loader.
# window_sizes=(12)
# seeds=(123 345 2024)                             # Pilot; later: (42 123 345 2024 2026)

# dl_method="PCA"
# train_ratio=0.50                       # Conditional share of non-test samples
# validation_ratio=0.50                  # Conditional share of non-test samples
# test_ratio=0.90                        # Effective targets: 5%, 5%, 90%
# number_of_bands=15
# block_size=32
# spatial_buffer=0
# search_trials=512
# batch_size=56
# epochs=50                               # Pilot; set to 50 for full experiments (no early stopping)

# # Test-only classification maps reuse the already computed test predictions.
# PREDICTION_MAP="${PREDICTION_MAP:-test}"
# # Use RUN_MODE=split_only to verify splitting before model training.
# RUN_MODE="${RUN_MODE:-train}"
# case "$RUN_MODE" in
#     train) extra_flags=(--prediction_map "$PREDICTION_MAP" --deterministic) ;;
#     split_only) extra_flags=(--verify_split_only) ;;
#     *) echo "Invalid RUN_MODE=$RUN_MODE; use train or split_only" >&2; exit 2 ;;
# esac

# if [[ ! -f "$PYTHON_SCRIPT" ]]; then
#     echo "Missing $PYTHON_SCRIPT in $(pwd). Set PYTHON_SCRIPT=main.py or put main.py here." >&2
#     exit 2
# fi

# LOG_DIR="logs"
# mkdir -p "$LOG_DIR"

# for dataset in "${datasets[@]}"; do
#     for window_size in "${window_sizes[@]}"; do
#         for seed in "${seeds[@]}"; do
#             stamp="$(date +%Y%m%d_%H%M%S)"
#             LOG_FILE="$LOG_DIR/${dataset}_ws${window_size}_seed${seed}_${stamp}.log"
#             echo "Dataset=$dataset WS=$window_size split_seed=$seed model_seed=$seed GPU=$GPU_IDS mode=$RUN_MODE"
#             echo "Logging to $LOG_FILE"

#             # Pipeline exit status is propagated by 'set -o pipefail'.
#             if PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
#                 python -u "$PYTHON_SCRIPT" \
#                     --dataset "$dataset" \
#                     --window_size "$window_size" \
#                     --dl_method "$dl_method" \
#                     --train_ratio "$train_ratio" \
#                     --validation_ratio "$validation_ratio" \
#                     --test_ratio "$test_ratio" \
#                     --number_of_bands "$number_of_bands" \
#                     --block_size "$block_size" \
#                     --spatial_buffer "$spatial_buffer" \
#                     --search_trials "$search_trials" \
#                     --batch_size "$batch_size" \
#                     --epochs "$epochs" \
#                     --seed "$seed" \
#                     --split_seed "$seed" \
#                     "${extra_flags[@]}" 2>&1 | tee "$LOG_FILE"; then
#                 echo "SUCCESS: $dataset WS=$window_size seed=$seed"
#             else
#                 rc=$?
#                 echo "FAILED (exit $rc): $dataset WS=$window_size seed=$seed. Inspect $LOG_FILE" >&2
#                 exit "$rc"
#             fi
#         done
#     done
# done

# echo "All requested experiments completed successfully. Logs: $LOG_DIR"











# #!/usr/bin/env bash
# # Spatially disjoint FKAN experiment launcher (for the seeded main file).
# set -Eeuo pipefail

# # main_seeded.py loads ./hsi_datasets relative to the current working directory.
# cd "$(dirname "${BASH_SOURCE[0]}")"

# export QT_QPA_PLATFORM=offscreen
# export MPLBACKEND=Agg
# export TF_FORCE_GPU_ALLOW_GROWTH=true
# export PYTHONUNBUFFERED=1

# PYTHON_SCRIPT="${PYTHON_SCRIPT:-main.py}"
# GPU_IDS="${GPU_IDS:-1}"                 # One GPU; two visible GPUs != distributed training.
# datasets=("HC")                        # Fe MUST be supported by dis_main.py's loader.
# window_sizes=(12)
# seeds=(42)                             # Pilot; later: (42 123 345 2024 2026)

# dl_method="PCA"
# train_ratio=0.50                       # Conditional share of non-test samples
# validation_ratio=0.50                  # Conditional share of non-test samples
# test_ratio=0.90                        # Effective targets: 5%, 5%, 90%
# number_of_bands=15
# block_size=32
# spatial_buffer=0
# search_trials=512
# batch_size=56
# epochs=3                              # Pilot; increase for the final experiment


# # Use RUN_MODE=split_only to verify splitting before model training.
# RUN_MODE="${RUN_MODE:-train}"
# case "$RUN_MODE" in
#     train) extra_flags=(--prediction_map full --deterministic) ;;
#     split_only) extra_flags=(--verify_split_only) ;;
#     *) echo "Invalid RUN_MODE=$RUN_MODE; use train or split_only" >&2; exit 2 ;;
# esac

# if [[ ! -f "$PYTHON_SCRIPT" ]]; then
#     echo "Missing $PYTHON_SCRIPT in $(pwd). Set PYTHON_SCRIPT=main_seeded.py or put dis_main.py here." >&2
#     exit 2
# fi

# LOG_DIR="logs"
# mkdir -p "$LOG_DIR"

# for dataset in "${datasets[@]}"; do
#     for window_size in "${window_sizes[@]}"; do
#         for seed in "${seeds[@]}"; do
#             stamp="$(date +%Y%m%d_%H%M%S)"
#             LOG_FILE="$LOG_DIR/${dataset}_ws${window_size}_seed${seed}_${stamp}.log"
#             echo "Dataset=$dataset WS=$window_size split_seed=$seed model_seed=$seed GPU=$GPU_IDS mode=$RUN_MODE"
#             echo "Logging to $LOG_FILE"

#             # Pipeline exit status is propagated by 'set -o pipefail'.
#             if PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
#                 python -u "$PYTHON_SCRIPT" \
#                     --dataset "$dataset" \
#                     --window_size "$window_size" \
#                     --dl_method "$dl_method" \
#                     --train_ratio "$train_ratio" \
#                     --validation_ratio "$validation_ratio" \
#                     --test_ratio "$test_ratio" \
#                     --number_of_bands "$number_of_bands" \
#                     --block_size "$block_size" \
#                     --spatial_buffer "$spatial_buffer" \
#                     --search_trials "$search_trials" \
#                     --batch_size "$batch_size" \
#                     --epochs "$epochs" \
#                     --seed "$seed" \
#                     --split_seed "$seed" \
#                     "${extra_flags[@]}" 2>&1 | tee "$LOG_FILE"; then
#                 echo "SUCCESS: $dataset WS=$window_size seed=$seed"
#             else
#                 rc=$?
#                 echo "FAILED (exit $rc): $dataset WS=$window_size seed=$seed. Inspect $LOG_FILE" >&2
#                 exit "$rc"
#             fi
#         done
#     done
# done

# echo "All requested experiments completed successfully. Logs: $LOG_DIR"







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

