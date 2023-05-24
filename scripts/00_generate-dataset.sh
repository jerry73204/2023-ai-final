#!/usr/bin/env bash
set -e

n_puzzles=100

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

cd "$script_dir/.."
mkdir dataset/puzzles

cd "$script_dir/../puzzle-generator"
poetry install

for (( pid=0; pid<$n_puzzles; pid++ )); do
    echo "poetry run main -- ../dataset/puzzles/$(printf 'p%04d' $pid)"
done | parallel

{
    echo '{'
    echo '  "puzzles": ['

    for (( pid=0; pid<$n_puzzles; pid++ )); do
        name=$(printf 'p%04d' $pid)

        echo '    {'
        echo "      \"name\": \"${name}\","
        echo "      \"n_pieces\": 25,"
        echo '    },'
    done

    echo '  ]'
    echo '}'
} > "$script_dir/../dataset/puzzles/manifest.json5"
