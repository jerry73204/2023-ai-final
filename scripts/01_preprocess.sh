#!/usr/bin/env bash
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

cd "$script_dir/../preprocess"

poetry install
rm -rf ../dataset/preprocess
poetry run main ../dataset/orig ../dataset/preprocess
