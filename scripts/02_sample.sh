#!/usr/bin/env bash
set -e
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
cd "${script_dir}/../jigsaw-diffusion"

model_path=$(ls "${script_dir}"/../log/model*.pt | sort -r | head -n1)

if [ -z "$model_path" ]; then
    echo "No model file found."
    exit 1
fi

echo "Using model file ${model_path}"

export OPENAI_LOGDIR="${script_dir}/../log"
poetry install
poetry run sample -- --data_dir ../dataset/puzzles --model_path="${model_path}" $@
