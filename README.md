# Solve Jigsaw Puzzle Using Diffsion Model

This project explores the ways to solve Jigsaw puzzle using using
diffusion model. It was initiated for the final project for Artificial
Intellegence course in Depart. of CSIE, National Taiwan University in
2023 spring.

## Related Works


- [*JigsawPlan: Room Layout Jigsaw Puzzle Extreme Structure from Motion using Diffusion Models*](https://arxiv.org/abs/2211.13785), Hosseini et al., 2022

## Download Dataset

The [Jigsaw
Puzzle](https://www.kaggle.com/datasets/etaifour/jigsawpuzzle) on
Kaggle is used in our project. Install
[kaggle](https://github.com/Kaggle/kaggle-api) command and have your
Kaggle credential configured on your system.

Run the script to download the dataset.

```sh
./scripts/00_download-dataset.sh
```

After the script is executed, the `dataset` directory looks like this.

```
dataset/
└── orig/
    ├── 10_p1.jpg
    ├── 10_p2.jpg
    ├── ...
    └── 9_p2.jpg
```

### Preprocess Dataset

Run the script to preprocess the dataset to extract puzzle piece
images.

```sh
./scripts/00_preprocess.sh
```

After the execution, the dataset directory looks like this.

```
dataset/
├── orig/
└── preprocess/
    ├── 10_p1/
    │   ├── 00.csv
    │   ├── 00.png
    │   ├── 01.csv
    │   ├── 01.png
    │   └── ...
    ├── 10_p1/
    │   ├── 00.csv
    │   ├── 00.png
    │   └── ...
    ├── ...
    └── 9_p2
        ├── 00.csv
        ├── 00.png
        └── ...
```
