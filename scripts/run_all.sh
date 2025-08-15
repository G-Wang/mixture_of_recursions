#!/bin/bash
set -e

# Orchestrate data download and pretraining on an arbitrary number of GPUs.
# Usage:
#   bash scripts/run_all.sh [accelerate|deepspeed] [online|offline] <num_gpus> (--model-size SIZE | <config1> [config2 ...])

launcher_type=""
run_mode=""

if [[ "$1" == "deepspeed" || "$1" == "accelerate" ]]; then
  launcher_type="$1"
  shift
fi

if [[ "$1" == "online" || "$1" == "offline" ]]; then
  run_mode="$1"
  shift
fi

if [ $# -lt 2 ]; then
  echo "Usage: $0 [accelerate|deepspeed] [online|offline] <num_gpus> (--model-size SIZE | <config1> [config2 ...])"
  exit 1
fi

num_gpus="$1"
shift

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]]; then
  echo "ERROR: <num_gpus> must be a positive integer"
  exit 1
fi

declare -a configs=()
if [[ "$1" == "--model-size" ]]; then
  shift
  model_size="$1"
  shift
  mapfile -t found_configs < <(ls conf/pretrain/*"${model_size}"*.yaml 2>/dev/null || true)
  for f in "${found_configs[@]}"; do
    configs+=("$(basename "$f" .yaml)")
  done
else
  configs=("$@")
fi

if [ ${#configs[@]} -eq 0 ]; then
  echo "ERROR: No config names found. Provide --model-size or explicit config names."
  exit 1
fi

# Step 0: create data directories if missing
mkdir -p hf_cache hf_datasets hf_models results

# Step 1: ensure datasets are available
bash lm_dataset/download_scripts/download_langauge_modeling_datasets.sh

# Step 2: launch training
GPU_IDS=$(seq -s, 0 $((num_gpus-1)))
cmd=(bash scripts/pretrain.sh)
[ -n "$launcher_type" ] && cmd+=("$launcher_type")
[ -n "$run_mode" ] && cmd+=("$run_mode")
cmd+=("$GPU_IDS")
for cfg in "${configs[@]}"; do
  "${cmd[@]}" "$cfg"
done
