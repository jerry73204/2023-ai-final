#!/usr/bin/env bash
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
cd "$script_dir"

mkdir -p ../dataset/orig
kaggle datasets download etaifour/jigsawpuzzle --unzip --path ../dataset/orig
