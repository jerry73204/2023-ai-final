# Jigsaw Diffusion Model

This directory provides the implementation of a diffusion model that
solves the Jigsaw puzzle. It is based on OpenAI's
[guided-diffusion](https://github.com/openai/guided-diffusion)
project.


## Installation

This project uses [poetry](https://python-poetry.org/docs/) to manage
depedencies and runtime environment. Please install poetry before
getting started.

To initialize the runtime environment,

```bash
poetry install
```

## Usage

To train a model,

```bash
poetry run train -- --data_dir ../dataset/example/
```
