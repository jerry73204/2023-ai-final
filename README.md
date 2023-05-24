# Solve Jigsaw Puzzle Using Diffsion Model

This project explores the ways to solve Jigsaw puzzle using using
diffusion model. It was initiated for the final project for Artificial
Intellegence course in Depart. of CSIE, National Taiwan University in
2023 spring.

## Related Works


- [*JigsawPlan: Room Layout Jigsaw Puzzle Extreme Structure from Motion using Diffusion Models*](https://arxiv.org/abs/2211.13785), Hosseini et al., 2022


## Prerequisites

- _poetry_, a dependency management tool for Python. Please follow this
  [doc](https://python-poetry.org/docs/) to install poetry.
- _GNU parallel_, a parallel command executor used in our
  scripts. Ubuntu users can install this command by `apt install
  parallel`.

## Get Started

Change directory to `scripts/` as we'll run the scripts within.

```sh
cd ./scripts
```

Generate a ground truth puzzle dataset. It will create a dataset
located at `REPO/dataset/puzzles`.

```sh
./00_generate-dataset.sh
```

Next, train the model. The model file will be saved to
`log/modeXXXXXXX.pt`.

```sh
./01_train.sh
```

Run the script to load most recently saved checkpoint file to generate
puzzle solutions.

```sh
./02_sample.sh
```
