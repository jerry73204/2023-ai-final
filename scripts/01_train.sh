#!/usr/bin/env bash
set -e

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

cd "${script_dir}/../jigsaw-diffusion"

export OPENAI_LOGDIR="${script_dir}/../log"
poetry install
poetry run train -- --data_dir ../dataset/puzzles --use_checkpoint=True
