#!/usr/bin/env bash
# FKANMamba — original greedy-block disjoint protocol.
# No diagnostic/paper seed restrictions. Use SEEDS and MODEL_SEED to override.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

PYTHON_SCRIPT="${PYTHON_SCRIPT:-main.py}"
GPU_IDS="${GPU_IDS:-0}"
EPOCHS="${EPOCHS:-50}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
PREDICTION_MAP="${PREDICTION_MAP:-test}"
RUN_MODE="${RUN_MODE:-train}"     # split_only, train Check partition before training.

datasets=(HC) # PI QUH TH HH
window_sizes=(12)
seeds=(42) # Spatial split seeds (override: SEEDS="42 123 2024")
if [[ -n "${SEEDS:-}" ]]; then read -r -a seeds <<< "$SEEDS"; fi

dl_method=PCA
number_of_bands=15
batch_size=56

case "$RUN_MODE" in
  train) flags=(--deterministic --prediction_map "$PREDICTION_MAP") ;;
  split_only) flags=(--verify_split_only) ;;
  *) echo 'RUN_MODE must be train or split_only' >&2; exit 2 ;;
esac

[[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing Python script: $PYTHON_SCRIPT" >&2; exit 2; }
[[ -d ./hsi_datasets ]] || { echo 'Missing ./hsi_datasets directory' >&2; exit 2; }

mkdir -p logs
for dataset in "${datasets[@]}"; do
  for window_size in "${window_sizes[@]}"; do
    for split_seed in "${seeds[@]}"; do
      model_seed="${MODEL_SEED:-$split_seed}"
      stamp="$(date +%Y%m%d_%H%M%S)"
      log="logs/${dataset}_disjoint_ws${window_size}_split${split_seed}_model${model_seed}_${stamp}.log"
      echo "Dataset=$dataset WS=$window_size split_seed=$split_seed model_seed=$model_seed block=$BLOCK_SIZE GPU=$GPU_IDS run=$RUN_MODE Python=$PYTHON_SCRIPT"
      if PYTHONHASHSEED="$model_seed" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
         python -u "$PYTHON_SCRIPT" \
           --dataset "$dataset" --window_size "$window_size" \
           --dl_method "$dl_method" --number_of_bands "$number_of_bands" \
           --batch_size "$batch_size" --epochs "$EPOCHS" \
           --seed "$model_seed" --split_seed "$split_seed" \
           --split_protocol disjoint --block_size "$BLOCK_SIZE" \
           --train_ratio 0.50 --validation_ratio 0.50 --test_ratio 0.90 \
           --spatial_buffer 0 \
           "${flags[@]}" 2>&1 | tee "$log"; then
        echo "SUCCESS: split_seed=$split_seed model_seed=$model_seed; log=$log"
      else
        rc=$?
        echo "FAILED (exit $rc): split_seed=$split_seed model_seed=$model_seed; log=$log" >&2
        exit "$rc"
      fi
    done
  done
done
echo 'Completed FKANMamba experiment.'

#!/usr/bin/env bash
# # FKANMamba Reviewer 2 — SpikeSST-style greedy-block disjoint sampling.
# # Calls the existing --split_protocol disjoint in main.py.
# # No architecture changes; this is NOT the old strict three-way block splitter.
# set -Eeuo pipefail
# cd "$(dirname "${BASH_SOURCE[0]}")"

# export QT_QPA_PLATFORM=offscreen
# export MPLBACKEND=Agg
# export TF_FORCE_GPU_ALLOW_GROWTH=true
# export PYTHONUNBUFFERED=1

# datasets=(HC) #PI QUH TH HH
# window_sizes=(12)
# seeds=(123) # 42 123 2024 3407 777
# if [[ -n "${SEEDS:-}" ]]; then read -r -a seeds <<< "$SEEDS"; fi
# dl_method=PCA
# number_of_bands=15
# batch_size=56

# PYTHON_SCRIPT="${PYTHON_SCRIPT:-main.py}"
# GPU_IDS="${GPU_IDS:-0}"
# EPOCHS="${EPOCHS:-50}"
# BLOCK_SIZE="${BLOCK_SIZE:-32}"
# PREDICTION_MAP="${PREDICTION_MAP:-test}"
# RUN_MODE="${RUN_MODE:-train}"     # Check partition before training.

# case "$RUN_MODE" in
#   train) flags=(--deterministic --prediction_map "$PREDICTION_MAP") ;;
#   split_only) flags=(--verify_split_only) ;;
#   *) echo 'RUN_MODE must be train or split_only' >&2; exit 2 ;;
# esac

# [[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing Python script: $PYTHON_SCRIPT" >&2; exit 2; }
# [[ -d ./hsi_datasets ]] || { echo 'Missing ./hsi_datasets directory' >&2; exit 2; }
# [[ "$BLOCK_SIZE" =~ ^[1-9][0-9]*$ ]] || {
#   echo 'BLOCK_SIZE must be a positive integer' >&2; exit 2;
# }

# mkdir -p logs
# for dataset in "${datasets[@]}"; do
#   for window_size in "${window_sizes[@]}"; do
#     for seed in "${seeds[@]}"; do
#       stamp="$(date +%Y%m%d_%H%M%S)"
#       log="logs/${dataset}_disjoint_ws${window_size}_seed${seed}_${stamp}.log"
#       echo "Protocol=Disjoint Dataset=$dataset WS=$window_size seed=$seed block=$BLOCK_SIZE GPU=$GPU_IDS mode=$RUN_MODE"
#       if PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
#          python -u "$PYTHON_SCRIPT" \
#            --dataset "$dataset" --window_size "$window_size" \
#            --dl_method "$dl_method" --number_of_bands "$number_of_bands" \
#            --batch_size "$batch_size" --epochs "$EPOCHS" \
#            --seed "$seed" --split_seed "$seed" \
#            --split_protocol disjoint --block_size "$BLOCK_SIZE" \
#            --train_ratio 0.50 --validation_ratio 0.50 --test_ratio 0.90 \
#            --spatial_buffer 0 \
#            "${flags[@]}" 2>&1 | tee "$log"; then
#         echo "SUCCESS: $dataset seed=$seed; log=$log"
#       else
#         rc=$?
#         echo "FAILED (exit $rc): $dataset seed=$seed; see $log" >&2
#         exit "$rc"
#       fi
#     done
#   done
# done
# echo 'Completed requested HC  disjoint experiments.'






 #!/usr/bin/env bash
# # FKANMamba Reviewer 2: patch-safe greedy-block spatial disjoint experiment.
# # Uses the EXISTING --split_protocol disjoint inside main.py (not SpikeSST model).
# set -Eeuo pipefail
# cd "$(dirname "${BASH_SOURCE[0]}")"

# export QT_QPA_PLATFORM=offscreen
# export MPLBACKEND=Agg
# export TF_FORCE_GPU_ALLOW_GROWTH=true
# export PYTHONUNBUFFERED=1

# # Initial HC pilot. For reviewer mean +/- std, supply SEEDS="42 123 2024 3407 777".
# datasets=(HC)
# window_sizes=(12)
# seeds=(123 2024 3407 777)
# if [[ -n "${SEEDS:-}" ]]; then read -r -a seeds <<< "$SEEDS"; fi

# dl_method=PCA
# number_of_bands=15
# batch_size=56

# PYTHON_SCRIPT="${PYTHON_SCRIPT:-main.py}"
# GPU_IDS="${GPU_IDS:-0}"
# EPOCHS="${EPOCHS:-50}"          # Pilot only. Use EPOCHS=50 for reviewer experiments.
# SEARCH_TRIALS="${SEARCH_TRIALS:-512}"
# SPATIAL_BUFFER="${SPATIAL_BUFFER:-0}"  # Zero extra buffer: FULL patch containment still enforced.
# BLOCK_SIZE="${BLOCK_SIZE:-32}"
# PREDICTION_MAP="${PREDICTION_MAP:-test}"
# RUN_MODE="${RUN_MODE:-split_only}"     # Check partition before training.

# case "$RUN_MODE" in
#   train) flags=(--deterministic --prediction_map "$PREDICTION_MAP") ;;
#   split_only) flags=(--verify_split_only) ;;
#   *) echo 'RUN_MODE must be train or split_only' >&2; exit 2 ;;
# esac

# [[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing Python script: $PYTHON_SCRIPT" >&2; exit 2; }
# [[ -d ./hsi_datasets ]] || { echo 'Missing ./hsi_datasets directory' >&2; exit 2; }
# [[ "$SPATIAL_BUFFER" =~ ^[0-9]+$ && "$BLOCK_SIZE" =~ ^[0-9]+$ ]] || {
#   echo 'BLOCK_SIZE and SPATIAL_BUFFER must be nonnegative integers' >&2; exit 2;
# }

# mkdir -p logs
# for dataset in "${datasets[@]}"; do
#   for window_size in "${window_sizes[@]}"; do
#     if (( BLOCK_SIZE < window_size + 2 * SPATIAL_BUFFER )); then
#       echo "Invalid geometry: block=$BLOCK_SIZE < window=$window_size + 2*buffer=$((2 * SPATIAL_BUFFER))." >&2
#       echo 'Use SPATIAL_BUFFER=0 for strict non-overlapping patch footprints with block 32.' >&2
#       exit 2
#     fi
#     for seed in "${seeds[@]}"; do
#       stamp="$(date +%Y%m%d_%H%M%S)"
#       log="logs/${dataset}_disjoint_ws${window_size}_seed${seed}_${stamp}.log"
#       echo "Protocol=disjoint Dataset=$dataset WS=$window_size seed=$seed block=$BLOCK_SIZE buffer=$SPATIAL_BUFFER GPU=$GPU_IDS mode=$RUN_MODE"
#       if PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
#          python -u "$PYTHON_SCRIPT" \
#            --dataset "$dataset" --window_size "$window_size" \
#            --dl_method "$dl_method" --number_of_bands "$number_of_bands" \
#            --batch_size "$batch_size" --epochs "$EPOCHS" \
#            --seed "$seed" --split_seed "$seed" \
#            --split_protocol disjoint --block_size "$BLOCK_SIZE" \
#            --train_ratio 0.50 --validation_ratio 0.50 --test_ratio 0.90 \
#            --search_trials "$SEARCH_TRIALS" --spatial_buffer "$SPATIAL_BUFFER" \
#            "${flags[@]}" 2>&1 | tee "$log"; then
#         echo "SUCCESS: $dataset seed=$seed; log=$log"
#       else
#         rc=$?
#         echo "FAILED (exit $rc): $dataset seed=$seed; see $log" >&2
#         exit "$rc"
#       fi
#     done
#   done
# done
# echo 'Completed requested greedy-block disjoint experiments.'


#!/usr/bin/env bash
# # Reproducible Reviewer 2 evaluation; no changes to FKANMamba or baseline architecture.
# set -Eeuo pipefail
# cd "$(dirname "${BASH_SOURCE[0]}")"
# export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1

# GPU_IDS="${GPU_IDS:-1}"
# DATA_DIR="${DATA_DIR:-./hsi_datasets}"
# DATASETS="${DATASETS:-PI QUH TH HC HH}"
# SEEDS="${SEEDS:-42 123 2024 3407 777}"
# WINDOW_SIZE="${WINDOW_SIZE:-12}"
# DL_METHOD="${DL_METHOD:-PCA}"
# NUMBER_OF_BANDS="${NUMBER_OF_BANDS:-15}"
# BATCH_SIZE="${BATCH_SIZE:-56}"
# EPOCHS="${EPOCHS:-50}"
# TRAIN_RATIO="${TRAIN_RATIO:-0.50}"
# VALIDATION_RATIO="${VALIDATION_RATIO:-0.50}"
# TEST_RATIO="${TEST_RATIO:-0.90}"
# SPLIT_PROTOCOL="${SPLIT_PROTOCOL:-blocks}"
# RUN_MODE="${RUN_MODE:-train}"
# [[ "$SPLIT_PROTOCOL" == blocks || "$SPLIT_PROTOCOL" == connected ]] || { echo 'SPLIT_PROTOCOL must be blocks or connected' >&2; exit 2; }
# [[ "$RUN_MODE" == train || "$RUN_MODE" == split_only ]] || { echo 'RUN_MODE must be train or split_only' >&2; exit 2; }
# [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo 'EPOCHS must be a positive integer' >&2; exit 2; }
# [[ "$WINDOW_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'WINDOW_SIZE must be a positive integer' >&2; exit 2; }
# read -r -a datasets <<< "$DATASETS"
# read -r -a seeds <<< "$SEEDS"
# (( ${#datasets[@]} > 0 && ${#seeds[@]} > 0 )) || { echo 'Specify at least one dataset and seed' >&2; exit 2; }
# for seed in "${seeds[@]}"; do [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Bad SEED=$seed" >&2; exit 2; }; done
# mkdir -p logs

# PYTHON_SCRIPT="${PYTHON_SCRIPT:-main.py}"
# BLOCK_SIZE="${BLOCK_SIZE:-32}"
# SPATIAL_BUFFER="${SPATIAL_BUFFER:-0}"
# SEARCH_TRIALS="${SEARCH_TRIALS:-512}"
# PREDICTION_MAP="${PREDICTION_MAP:-test}"
# [[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing $PYTHON_SCRIPT" >&2; exit 2; }
# [[ -d "$DATA_DIR" ]] || { echo "Missing $DATA_DIR" >&2; exit 2; }
# if [[ "$RUN_MODE" == train ]]; then flags=(--deterministic --prediction_map "$PREDICTION_MAP");
# else flags=(--verify_split_only); fi
# # main.py reads ./hsi_datasets; keep this agreed project directory.
# if [[ "$DATA_DIR" != ./hsi_datasets ]]; then
#   echo 'Current FKAN main.py requires DATA_DIR=./hsi_datasets' >&2; exit 2
# fi
# for dataset in "${datasets[@]}"; do
#   for seed in "${seeds[@]}"; do
#     logfile="logs/fkan_${dataset}_${SPLIT_PROTOCOL}_ws${WINDOW_SIZE}_split${seed}_model${seed}_$(date +%Y%m%d_%H%M%S).log"
#     echo "FKANMamba dataset=$dataset split_protocol=$SPLIT_PROTOCOL block_size=$BLOCK_SIZE WS=$WINDOW_SIZE split_seed=$seed model_seed=$seed mode=$RUN_MODE"
#     PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
#       python -u "$PYTHON_SCRIPT" \
#         --dataset "$dataset" --window_size "$WINDOW_SIZE" --dl_method "$DL_METHOD" \
#         --number_of_bands "$NUMBER_OF_BANDS" --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" \
#         --train_ratio "$TRAIN_RATIO" --validation_ratio "$VALIDATION_RATIO" --test_ratio "$TEST_RATIO" \
#         --split_protocol "$SPLIT_PROTOCOL" --block_size "$BLOCK_SIZE" \
#         --spatial_buffer "$SPATIAL_BUFFER" --search_trials "$SEARCH_TRIALS" \
#         --seed "$seed" --split_seed "$seed" "${flags[@]}" 2>&1 | tee "$logfile"
#   done
# done
