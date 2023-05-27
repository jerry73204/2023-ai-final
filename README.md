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
- `optipng`. Ubuntu users can install by `apt install optipng`.
- `rsvg-convert`. Ubuntu users can install by `apt install librsvg2-bin`.
- `potrace`. Ubuntu users can install by `apt install potrace`.
- `git-lfs` is used to manage large weights files. Read [this
  page](https://github.com/git-lfs/git-lfs#installing) to install it.


## Prepare the Repository

After this repo is freshly cloned, initialize the environment.

```sh
git lfs install
```

and initialize submodules.

```sh
git submodule update --init --recursive
```

## Get Started

Change directory to `scripts/` as we'll run the scripts within.

```sh
cd ./scripts
```

### Dataset Generation

Generate a ground truth puzzle dataset. It will create a dataset
located at `REPO/dataset/puzzles`.

```sh
./00_generate-dataset.sh
```

### Training

Next, train the model. The weights file will be saved to
`log/modeXXXXXXX.pt`.


```sh
./01_train.sh
```

The weights files can be saved via `git-lfs`. Please read the later
[Manage Large Weights Files](#manage-large-weights-files) for more
details.


### Testing

Run the script to load most recently saved checkpoint file to generate
puzzle solutions.

```sh
./02_sample.sh
```

## Manage Large Weights Files

Large weights files are usually stored in a separate
`exp/<date>/<comment>` branch other than `main`. For example,

```sh
git checkout exp/2023-05-27/no-agumentation
```

Use git-lfs to download large weights files in the `log/` directory.

```sh
git lfs pull
```

To upload new weights files,

```sh
git add log/
git commit -m 'Save weights files.'
git push
```
