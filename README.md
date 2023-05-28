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

Run the command below to train the model. The weights file will be
saved to `log/modeXXXXXXX.pt`. The weights files can be saved to
`log/` directory using `git-lfs`. Please read the later [Manage Large
Weights Files](#manage-large-weights-files) for more details.

```sh
./01_train.sh
```

Pass command line arguments to tweak the batch size. The default batch
size is 1.

```sh
./01_train.sh --batch_size=16
```

Pass `--show_gui` to show the input puzzles fed into the model during
training.

```sh
./01_train.sh --show_gui
```

Pass `--resume_checkpoint` to resume the model weights from a weights
file.

```sh
./01_train.sh --resume_checkpoint=$PWD/../log/model050000.pt
```

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

## Fine-Tuning

In the file `jigsaw-diffusion/jigsaw_diffusion/jigsaw_script_util.py`,
tweak the model parameters here.

```python
def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=500,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
```

In the file `jigsaw-diffusion/jigsaw_diffusion/train.py`,
tweak the training parameters here.

```python
def parse_args():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        piece_size=64,
        puzzle_size=320,
    )
```
