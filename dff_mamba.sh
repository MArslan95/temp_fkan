#!/usr/bin/env bash
# Reproducible Reviewer 2 evaluation; no changes to FKANMamba or baseline architecture.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1

GPU_IDS="${GPU_IDS:-0}"
DATA_DIR="${DATA_DIR:-./hsi_datasets}"
DATASETS="${DATASETS:-PI QUH TH HC HH}"
SEEDS="${SEEDS:-42 123 2024 3407 777}"
WINDOW_SIZE="${WINDOW_SIZE:-12}"
DL_METHOD="${DL_METHOD:-PCA}"
NUMBER_OF_BANDS="${NUMBER_OF_BANDS:-15}"
BATCH_SIZE="${BATCH_SIZE:-56}"
EPOCHS="${EPOCHS:-50}"
TRAIN_RATIO="${TRAIN_RATIO:-0.50}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.50}"
TEST_RATIO="${TEST_RATIO:-0.90}"
SPLIT_PROTOCOL="${SPLIT_PROTOCOL:-blocks}"
RUN_MODE="${RUN_MODE:-train}"
[[ "$SPLIT_PROTOCOL" == blocks || "$SPLIT_PROTOCOL" == connected ]] || { echo 'SPLIT_PROTOCOL must be blocks or connected' >&2; exit 2; }
[[ "$RUN_MODE" == train || "$RUN_MODE" == split_only ]] || { echo 'RUN_MODE must be train or split_only' >&2; exit 2; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo 'EPOCHS must be a positive integer' >&2; exit 2; }
[[ "$WINDOW_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'WINDOW_SIZE must be a positive integer' >&2; exit 2; }
read -r -a datasets <<< "$DATASETS"
read -r -a seeds <<< "$SEEDS"
(( ${#datasets[@]} > 0 && ${#seeds[@]} > 0 )) || { echo 'Specify at least one dataset and seed' >&2; exit 2; }
for seed in "${seeds[@]}"; do [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Bad SEED=$seed" >&2; exit 2; }; done
mkdir -p logs

PYTHON_SCRIPT="${PYTHON_SCRIPT:-baseline_mamba.py}"
MODELS="${MODELS:-MulMamba}" # MorpMamba SSFormer Hybrid_ViT DSST WaveMamba
FKAN_PROJECT_DIR="${FKAN_PROJECT_DIR:-$(pwd)}"
MAIN_RESULTS_ROOT="${MAIN_RESULTS_ROOT:-KanMamba_8_chnal_32_bs_${BATCH_SIZE}}"
MAIN_ARCH_TAG="${MAIN_ARCH_TAG:-original}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-}"
read -r -a models <<< "$MODELS"
(( ${#models[@]} > 0 )) || { echo 'Empty MODELS' >&2; exit 2; }
for model in "${models[@]}"; do
  case "$model" in MorpMamba|SSFormer|Hybrid_ViT|DSST|WaveMamba|MulMamba) ;; *) echo "Unknown baseline $model" >&2; exit 2;; esac
done
[[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing $PYTHON_SCRIPT" >&2; exit 2; }
[[ -d "$DATA_DIR" ]] || { echo "Missing $DATA_DIR" >&2; exit 2; }
if [[ "$TEST_RATIO" != 0.90 && "$TEST_RATIO" != 0.9 ]]; then
  echo 'Baseline result directory convention requires TEST_RATIO=0.90' >&2; exit 2
fi
if [[ -n "$REFERENCE_SPLIT" ]] && (( ${#datasets[@]} != 1 || ${#seeds[@]} != 1 )); then
  echo 'REFERENCE_SPLIT may only be used with one dataset and one seed' >&2; exit 2
fi
if [[ "$RUN_MODE" == split_only ]]; then flags=(--verify_split_only); models=("${models[0]}");
else flags=(--deterministic); fi
for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    if [[ -n "$REFERENCE_SPLIT" ]]; then split_file="$REFERENCE_SPLIT";
    else
      if [[ "$SPLIT_PROTOCOL" == blocks ]]; then split_dir="split_${seed}";
      else split_dir="split_${seed}_connected_patchsafe"; fi
      split_file="$FKAN_PROJECT_DIR/$MAIN_RESULTS_ROOT/$dataset/$WINDOW_SIZE/$MAIN_ARCH_TAG/$split_dir/seed_${seed}/disjoint_spatial_split.npz"
    fi
    [[ -f "$split_file" ]] || { echo "Missing FKAN $SPLIT_PROTOCOL split: $split_file" >&2; exit 2; }
    for model in "${models[@]}"; do
      logfile="logs/baseline_${model}_${dataset}_${SPLIT_PROTOCOL}_ws${WINDOW_SIZE}_split${seed}_model${seed}_$(date +%Y%m%d_%H%M%S).log"
      echo "Baseline=$model Dataset=$dataset Protocol=$SPLIT_PROTOCOL WS=$WINDOW_SIZE split_seed=$seed model_seed=$seed"
      echo "Reference split=$split_file"
      if PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
         python -u "$PYTHON_SCRIPT" \
           --model "$model" --dataset "$dataset" --window_size "$WINDOW_SIZE" \
           --dl_method "$DL_METHOD" --number_of_bands "$NUMBER_OF_BANDS" \
           --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" \
           --train_ratio "$TRAIN_RATIO" --validation_ratio "$VALIDATION_RATIO" --test_ratio "$TEST_RATIO" \
           --seed "$seed" --split_seed "$seed" --data_dir "$DATA_DIR" \
           --reference_split "$split_file" --expected_split_protocol "$SPLIT_PROTOCOL" \
           "${flags[@]}" 2>&1 | tee "$logfile"; then
         echo "SUCCESS $model / $dataset / $SPLIT_PROTOCOL / seed=$seed"
         if [[ "$RUN_MODE" == train ]]; then
           output_root="baseline_32_8_${BATCH_SIZE}/${dataset}/0.9/${WINDOW_SIZE}/split_${SPLIT_PROTOCOL}_${seed}/seed_${seed}"
           summary="$output_root/all_models_results.json"
           [[ -f "$summary" ]] || { echo "Missing model summary: $summary" >&2; exit 2; }
           cp "$summary" "$output_root/all_models_results_${model}.json"
         fi
       else
         echo "FAILED: $model / $dataset / $seed; inspect $logfile" >&2
         exit 1
       fi
    done
  done
done













# #!/usr/bin/env bash
# # Train all five unchanged baselines on FKANMamba's EXACT SAVED spatial partition.
# # No baseline creates or changes the train/validation/test split.
# set -Eeuo pipefail
# cd "$(dirname "${BASH_SOURCE[0]}")"

# export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
# export TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1

# PYTHON_SCRIPT="${PYTHON_SCRIPT:-baseline_mamba.py}"
# GPU_IDS="${GPU_IDS:-1}"
# DATA_DIR="${DATA_DIR:-./hsi_datasets}"
# RUN_MODE="${RUN_MODE:-train}"             # split_only | train
# FKAN_PROJECT_DIR="${FKAN_PROJECT_DIR:-$HOME/Desktop/temp_fkan-main}"
# MAIN_RESULTS_ROOT="${MAIN_RESULTS_ROOT:-KanMamba_8_chnal_32_bs_56}"
# MAIN_ARCH_TAG="${MAIN_ARCH_TAG:-arch_original}"
# # Matches the exact folder written by the current connected-region FKANMamba main.py.
# # For other saved naming schemes set SPLIT_FOLDER explicitly.
# SPLIT_FOLDER="${SPLIT_FOLDER:-}"
# REFERENCE_SPLIT="${REFERENCE_SPLIT:-}"   # Optional explicit saved .npz (single dataset/split)
# SPLIT_SEED="${SPLIT_SEED:-}"             # Optional fixed split seed; otherwise matches model seed
# DATASETS="${DATASETS:-HC HH}"
# SEEDS="${SEEDS:-42}"
# WINDOW_SIZES="${WINDOW_SIZES:-12}"
# MODELS="${MODELS:-MorpMamba SSFormer Hybrid_ViT DSST WaveMamba}"
# EPOCHS="${EPOCHS:-50}"
# BATCH_SIZE="${BATCH_SIZE:-56}"
# NUMBER_OF_BANDS="${NUMBER_OF_BANDS:-15}"
# DL_METHOD="${DL_METHOD:-PCA}"

# if [[ "$DATASETS" == ALL ]]; then
#   DATASETS="LK HC UH13 PU PC SA HH QUH PI TH BS"
# fi
# read -r -a datasets <<< "$DATASETS"
# read -r -a seeds <<< "$SEEDS"
# read -r -a window_sizes <<< "$WINDOW_SIZES"
# read -r -a models <<< "$MODELS"
# for name in DATASETS SEEDS WINDOW_SIZES MODELS; do
#   case "$name" in
#     DATASETS) (( ${#datasets[@]} > 0 )) || { echo 'Empty DATASETS' >&2; exit 2; } ;;
#     SEEDS) (( ${#seeds[@]} > 0 )) || { echo 'Empty SEEDS' >&2; exit 2; } ;;
#     WINDOW_SIZES) (( ${#window_sizes[@]} > 0 )) || { echo 'Empty WINDOW_SIZES' >&2; exit 2; } ;;
#     MODELS) (( ${#models[@]} > 0 )) || { echo 'Empty MODELS' >&2; exit 2; } ;;
#   esac
# done
# [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo 'EPOCHS must be positive' >&2; exit 2; }
# for seed in "${seeds[@]}"; do
#   [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid SEED $seed" >&2; exit 2; }
# done
# for model in "${models[@]}"; do
#   case "$model" in
#     MorpMamba|SSFormer|Hybrid_ViT|DSST|WaveMamba) ;;
#     *) echo "Unsupported model: $model" >&2; exit 2 ;;
#   esac
# done
# case "$RUN_MODE" in
#   train) flags=(--deterministic) ;;
#   split_only) flags=(--verify_split_only) ;;
#   *) echo 'RUN_MODE must be train or split_only' >&2; exit 2 ;;
# esac
# [[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing baseline script: $PYTHON_SCRIPT" >&2; exit 2; }
# [[ -d "$DATA_DIR" ]] || { echo "Missing dataset folder: $DATA_DIR" >&2; exit 2; }
# mkdir -p logs

# # A shared partition only needs one check. All five models read it unchanged.
# if [[ "$RUN_MODE" == split_only ]]; then
#   models=("${models[0]}")
# fi

# failures=()
# for dataset in "${datasets[@]}"; do
#   for window_size in "${window_sizes[@]}"; do
#     for seed in "${seeds[@]}"; do
#       split_seed="${SPLIT_SEED:-$seed}"
#       if [[ -n "$REFERENCE_SPLIT" ]]; then
#         split_file="$REFERENCE_SPLIT"
#       else
#         folder="${SPLIT_FOLDER:-split_${split_seed}_github_connected_patchsafe}"
#         split_file="$FKAN_PROJECT_DIR/$MAIN_RESULTS_ROOT/$dataset/$window_size/$MAIN_ARCH_TAG/$folder/seed_${split_seed}/disjoint_spatial_split.npz"
#       fi
#       if [[ ! -f "$split_file" ]]; then
#         echo "Missing FKAN connected split: $split_file" >&2
#         echo 'Run main.py with --split_protocol connected for this dataset/split seed first.' >&2
#         exit 2
#       fi
#       for model in "${models[@]}"; do
#         stamp="$(date +%Y%m%d_%H%M%S)"
#         logfile="logs/baseline_${model}_${dataset}_ws${window_size}_split${split_seed}_seed${seed}_${stamp}.log"
#         echo "Model=$model Dataset=$dataset WS=$window_size split_seed=$split_seed model_seed=$seed GPU=$GPU_IDS mode=$RUN_MODE"
#         echo "Reference partition: $split_file"
#         if PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
#           python -u "$PYTHON_SCRIPT" \
#             --model "$model" --dataset "$dataset" \
#             --window_size "$window_size" --dl_method "$DL_METHOD" \
#             --number_of_bands "$NUMBER_OF_BANDS" --batch_size "$BATCH_SIZE" \
#             --epochs "$EPOCHS" --seed "$seed" --split_seed "$split_seed" \
#             --data_dir "$DATA_DIR" --reference_split "$split_file" \
#             --expected_split_protocol connected \
#             "${flags[@]}" 2>&1 | tee "$logfile"; then
#           echo "SUCCESS: $model / $dataset / split=$split_seed / seed=$seed"
#           if [[ "$RUN_MODE" == train ]]; then
#             output_root="baseline_32_8_${BATCH_SIZE}/${dataset}/0.9/${window_size}/split_connected_${split_seed}/seed_${seed}"
#             summary="$output_root/all_models_results.json"
#             if [[ -f "$summary" ]]; then
#               archived="$output_root/all_models_results_${model}.json"
#               cp "$summary" "$archived"
#               echo "Saved: $archived"
#             else
#               echo "WARNING: no model results summary: $summary" >&2
#             fi
#           fi
#         else
#           rc=$?
#           echo "FAILED: $model / $dataset / seed=$seed (exit $rc); see $logfile" >&2
#           failures+=("$model/$dataset/$seed (exit $rc)")
#         fi
#       done
#     done
#   done
# done
# if (( ${#failures[@]} )); then
#   echo "${#failures[@]} baseline run(s) failed: ${failures[*]}" >&2
#   exit 1
# fi
# if [[ "$RUN_MODE" == split_only ]]; then
#   echo 'FKANMamba saved connected partitions verified. No baseline was trained.'
# else
#   echo 'All requested baseline runs completed.'
# fi







# #!/usr/bin/env bash
# # Run five ORIGINAL baselines sequentially against the EXACT saved FKANMamba
# # spatial split. Each architecture trains in its own Python/GPU process.
# # This launcher does not modify baseline model architectures or training code.
# set -Eeuo pipefail
# cd "$(dirname "${BASH_SOURCE[0]}")"

# export QT_QPA_PLATFORM=offscreen
# export MPLBACKEND=Agg
# export TF_FORCE_GPU_ALLOW_GROWTH=true
# export PYTHONUNBUFFERED=1

# PYTHON_SCRIPT="${PYTHON_SCRIPT:-baseline_mamba.py}"
# GPU_IDS="${GPU_IDS:-1}"
# DATA_DIR="${DATA_DIR:-./hsi_datasets}"
# RUN_MODE="${RUN_MODE:-train}"                   # train | split_only
# FKAN_PROJECT_DIR="${FKAN_PROJECT_DIR:-$HOME/Desktop/temp_fkan-main}"
# MAIN_RESULTS_ROOT="${MAIN_RESULTS_ROOT:-KanMamba_8_chnal_32_bs_56}"
# MAIN_ARCH_TAG="${MAIN_ARCH_TAG:-arch_original_dima_ssm_tokenfix}"
# SPLIT_SEED="${SPLIT_SEED:-42}"
# REFERENCE_SPLIT="${REFERENCE_SPLIT:-}"           # optional exact .npz path

# # All five requested baseline architectures; override with MODELS="SSFormer DSST".
# read -r -a models <<< "${MODELS:-MorpMamba SSFormer Hybrid_ViT DSST WaveMamba}" #{MODELS:-MorpMamba SSFormer Hybrid_ViT DSST WaveMamba}
# datasets=("HC")
# window_sizes=(12)
# seeds=(42)
# epochs="${EPOCHS:-50}"                           # smoke test; EPOCHS=50 for final runs
# batch_size=56
# number_of_bands=15
# dl_method="PCA"
# test_ratio=0.90
# train_ratio=0.50
# validation_ratio=0.50

# [[ "$epochs" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be a positive integer" >&2; exit 2; }
# (( ${#models[@]} > 0 )) || { echo "No baseline models selected" >&2; exit 2; }
# for model in "${models[@]}"; do
#   case "$model" in
#     MorpMamba|SSFormer|Hybrid_ViT|DSST|WaveMamba) ;;
#     *) echo "Unsupported model: '$model'" >&2; exit 2 ;;
#   esac
# done
# case "$RUN_MODE" in
#   train) extra_flags=(--deterministic) ;;
#   split_only) extra_flags=(--verify_split_only) ;;
#   *) echo 'RUN_MODE must be train or split_only' >&2; exit 2 ;;
# esac
# [[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing $PYTHON_SCRIPT in $(pwd)" >&2; exit 2; }
# [[ -d "$DATA_DIR" ]] || { echo "Missing dataset directory: $DATA_DIR" >&2; exit 2; }
# mkdir -p logs

# if [[ "$RUN_MODE" == split_only ]]; then
#   # The five baselines use the same saved coordinates; check them just once.
#   models=("${models[0]}")
#   echo "Split-only: verify the shared partition once, without constructing a model."
# fi

# failures=()
# for dataset in "${datasets[@]}"; do
#   for window_size in "${window_sizes[@]}"; do
#     if [[ -n "$REFERENCE_SPLIT" ]]; then
#       split_file="$REFERENCE_SPLIT"
#     else
#       split_file="$FKAN_PROJECT_DIR/$MAIN_RESULTS_ROOT/$dataset/$window_size/$MAIN_ARCH_TAG/split_${SPLIT_SEED}/seed_${SPLIT_SEED}/disjoint_spatial_split.npz"
#     fi
#     [[ -f "$split_file" ]] || { echo "Missing saved FKAN split: $split_file" >&2; exit 2; }

#     for seed in "${seeds[@]}"; do
#       for model in "${models[@]}"; do
#         stamp="$(date +%Y%m%d_%H%M%S)"
#         logfile="logs/baseline_${model}_${dataset}_ws${window_size}_split${SPLIT_SEED}_seed${seed}_${stamp}.log"
#         echo "------------------------------------------------------------"
#         echo "Baseline=$model Dataset=$dataset WS=$window_size Epochs=$epochs SplitSeed=$SPLIT_SEED ModelSeed=$seed GPU=$GPU_IDS"
#         echo "Data directory=$DATA_DIR | FKAN reference split=$split_file"
#         echo "Logging to $logfile"

#         if PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
#             python -u "$PYTHON_SCRIPT" \
#               --model "$model" --dataset "$dataset" \
#               --window_size "$window_size" --dl_method "$dl_method" \
#               --train_ratio "$train_ratio" --validation_ratio "$validation_ratio" --test_ratio "$test_ratio" \
#               --number_of_bands "$number_of_bands" --batch_size "$batch_size" \
#               --epochs "$epochs" --seed "$seed" --split_seed "$SPLIT_SEED" \
#               --data_dir "$DATA_DIR" --reference_split "$split_file" \
#               "${extra_flags[@]}" 2>&1 | tee "$logfile"; then
#           echo "SUCCESS: $model / $dataset / WS=$window_size / seed=$seed"
#           if [[ "$RUN_MODE" == train ]]; then
#             # baseline_mamba.py rewrites one shared all_models_results.json.
#             # Archive each finished model's JSON before starting the next model.
#             output_root="baseline_32_8_${batch_size}/${dataset}/0.9/${window_size}"
#             summary="$output_root/all_models_results.json"
#             if [[ -f "$summary" ]]; then
#               archived="$output_root/all_models_results_${model}_split${SPLIT_SEED}_seed${seed}_e${epochs}_${stamp}.json"
#               cp "$summary" "$archived"
#               echo "Saved model-specific summary: $archived"
#             else
#               echo "WARNING: no shared results summary found at $summary" >&2
#             fi
#           fi
#         else
#           rc=$?
#           echo "FAILED (exit $rc): $model — inspect $logfile" >&2
#           failures+=("$model / $dataset / WS=$window_size / seed=$seed (exit $rc): $logfile")
#           # Continue with the other requested baselines in independent processes.
#         fi
#       done
#     done
#   done
# done

# if (( ${#failures[@]} > 0 )); then
#   echo "${#failures[@]} baseline run(s) failed:" >&2
#   printf '  %s\n' "${failures[@]}" >&2
#   exit 1
# fi
# if [[ "$RUN_MODE" == split_only ]]; then
#   echo "Reference split verified; all five models will use this same partition."
# else
#   echo "All requested baseline experiments completed successfully. Logs: logs/"
# fi

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
