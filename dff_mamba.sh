#!/usr/bin/env bash
# Run five ORIGINAL baselines sequentially against the EXACT saved FKANMamba
# spatial split. Each architecture trains in its own Python/GPU process.
# This launcher does not modify baseline model architectures or training code.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

PYTHON_SCRIPT="${PYTHON_SCRIPT:-baseline_mamba.py}"
GPU_IDS="${GPU_IDS:-1}"
DATA_DIR="${DATA_DIR:-./hsi_datasets}"
RUN_MODE="${RUN_MODE:-train}"                   # train | split_only
FKAN_PROJECT_DIR="${FKAN_PROJECT_DIR:-$HOME/Desktop/temp_fkan-main}"
MAIN_RESULTS_ROOT="${MAIN_RESULTS_ROOT:-KanMamba_8_chnal_32_bs_56}"
MAIN_ARCH_TAG="${MAIN_ARCH_TAG:-arch_original_dima_ssm_tokenfix}"
SPLIT_SEED="${SPLIT_SEED:-42}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-}"           # optional exact .npz path

# All five requested baseline architectures; override with MODELS="SSFormer DSST".
read -r -a models <<< "${MODELS:-WaveMamba}" #{MODELS:-MorpMamba SSFormer Hybrid_ViT DSST WaveMamba}
datasets=("HC")
window_sizes=(12)
seeds=(42)
epochs="${EPOCHS:-50}"                           # smoke test; EPOCHS=50 for final runs
batch_size=56
number_of_bands=15
dl_method="PCA"
test_ratio=0.90
train_ratio=0.50
validation_ratio=0.50

[[ "$epochs" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be a positive integer" >&2; exit 2; }
(( ${#models[@]} > 0 )) || { echo "No baseline models selected" >&2; exit 2; }
for model in "${models[@]}"; do
  case "$model" in
    MorpMamba|SSFormer|Hybrid_ViT|DSST|WaveMamba) ;;
    *) echo "Unsupported model: '$model'" >&2; exit 2 ;;
  esac
done
case "$RUN_MODE" in
  train) extra_flags=(--deterministic) ;;
  split_only) extra_flags=(--verify_split_only) ;;
  *) echo 'RUN_MODE must be train or split_only' >&2; exit 2 ;;
esac
[[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing $PYTHON_SCRIPT in $(pwd)" >&2; exit 2; }
[[ -d "$DATA_DIR" ]] || { echo "Missing dataset directory: $DATA_DIR" >&2; exit 2; }
mkdir -p logs

if [[ "$RUN_MODE" == split_only ]]; then
  # The five baselines use the same saved coordinates; check them just once.
  models=("${models[0]}")
  echo "Split-only: verify the shared partition once, without constructing a model."
fi

failures=()
for dataset in "${datasets[@]}"; do
  for window_size in "${window_sizes[@]}"; do
    if [[ -n "$REFERENCE_SPLIT" ]]; then
      split_file="$REFERENCE_SPLIT"
    else
      split_file="$FKAN_PROJECT_DIR/$MAIN_RESULTS_ROOT/$dataset/$window_size/$MAIN_ARCH_TAG/split_${SPLIT_SEED}/seed_${SPLIT_SEED}/disjoint_spatial_split.npz"
    fi
    [[ -f "$split_file" ]] || { echo "Missing saved FKAN split: $split_file" >&2; exit 2; }

    for seed in "${seeds[@]}"; do
      for model in "${models[@]}"; do
        stamp="$(date +%Y%m%d_%H%M%S)"
        logfile="logs/baseline_${model}_${dataset}_ws${window_size}_split${SPLIT_SEED}_seed${seed}_${stamp}.log"
        echo "------------------------------------------------------------"
        echo "Baseline=$model Dataset=$dataset WS=$window_size Epochs=$epochs SplitSeed=$SPLIT_SEED ModelSeed=$seed GPU=$GPU_IDS"
        echo "Data directory=$DATA_DIR | FKAN reference split=$split_file"
        echo "Logging to $logfile"

        if PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
            python -u "$PYTHON_SCRIPT" \
              --model "$model" --dataset "$dataset" \
              --window_size "$window_size" --dl_method "$dl_method" \
              --train_ratio "$train_ratio" --validation_ratio "$validation_ratio" --test_ratio "$test_ratio" \
              --number_of_bands "$number_of_bands" --batch_size "$batch_size" \
              --epochs "$epochs" --seed "$seed" --split_seed "$SPLIT_SEED" \
              --data_dir "$DATA_DIR" --reference_split "$split_file" \
              "${extra_flags[@]}" 2>&1 | tee "$logfile"; then
          echo "SUCCESS: $model / $dataset / WS=$window_size / seed=$seed"
          if [[ "$RUN_MODE" == train ]]; then
            # baseline_mamba.py rewrites one shared all_models_results.json.
            # Archive each finished model's JSON before starting the next model.
            output_root="baseline_32_8_${batch_size}/${dataset}/0.9/${window_size}"
            summary="$output_root/all_models_results.json"
            if [[ -f "$summary" ]]; then
              archived="$output_root/all_models_results_${model}_split${SPLIT_SEED}_seed${seed}_e${epochs}_${stamp}.json"
              cp "$summary" "$archived"
              echo "Saved model-specific summary: $archived"
            else
              echo "WARNING: no shared results summary found at $summary" >&2
            fi
          fi
        else
          rc=$?
          echo "FAILED (exit $rc): $model — inspect $logfile" >&2
          failures+=("$model / $dataset / WS=$window_size / seed=$seed (exit $rc): $logfile")
          # Continue with the other requested baselines in independent processes.
        fi
      done
    done
  done
done

if (( ${#failures[@]} > 0 )); then
  echo "${#failures[@]} baseline run(s) failed:" >&2
  printf '  %s\n' "${failures[@]}" >&2
  exit 1
fi
if [[ "$RUN_MODE" == split_only ]]; then
  echo "Reference split verified; all five models will use this same partition."
else
  echo "All requested baseline experiments completed successfully. Logs: logs/"
fi

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
