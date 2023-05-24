# Solve Jigsaw Puzzle Using Diffsion Model

This project explores the ways to solve Jigsaw puzzle using using
diffusion model. It was initiated for the final project for Artificial
Intellegence course in Depart. of CSIE, National Taiwan University in
2023 spring.

## Related Works


- [*JigsawPlan: Room Layout Jigsaw Puzzle Extreme Structure from Motion using Diffusion Models*](https://arxiv.org/abs/2211.13785), Hosseini et al., 2022


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

To start model training,

```sh
./01_train.sh
```
