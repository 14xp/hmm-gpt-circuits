# GPT Circuits
Repository for training toy GPT-2 models and experimenting with sparse autoencoders.

## File Structure
* checkpoints - Training output
* [config](config) - Model and training configurations
* [data](data) - Datasets to be used for training
* [experiments](experiments) - Scripts for experiments
* [models](models) - GPT and SAE model definitions
* [training](training) - Model trainers

## Setup

### Environment
Python requirements are listed in [requirements.txt](requirements.txt). To install requirements, run:

```
pip install -r requirements.txt
```

### Datasets
Each dataset contains a [prepare.py](data/shakespeare/prepare.py) file. Run this file as a module from the root directory to prepare a dataset for use during training. Example:
```
python -m data.shakespeare.prepare
```

## Training

### GPT-2

Configurations are stored in [config/gpt](config/gpt). The trainer is located at [training/gpt.py](training/gpt.py). To run training, use:

```
python -m training.gpt --config=shakespeare_64x4
```

DDP is supported:

```
torchrun --standalone --nproc_per_node=8 -m training.gpt --config=shakespeare_64x4
```

### Sparse Autoencoders

Configurations are stored in [config/sae](config/sae). The Trainers are located at [training/sae](training/sae). To run training, use:

```
python -m training.sae.main --config=standard.shakespeare_64x4 --load_from=shakespeare_64x4
```

To run the staircase SAE, use:

```
python -m training.sae.main --config=topk-staircase-share.shakespeare_64x4 --load_from=shakespeare_64x4
python -m training.sae.main --config=topk-staircase-noshare.shakespeare_64x4 --load_from=shakespeare_64x4
```

To run the Jacobian over MLP Block (including LayerNorm)

```
python -m training.sae.main --load-from shakespeare_64x4 --config jsae.mlp_ln.shk_64x4 --sparsity 1e-2 
python -m training.sae.main --load-from gpt2 --config jln.mlpblock.gpt2 --sparsity 1e-6
```

### SAE location conventions

SAE keys are of the format `{layer_idx}_{hook_point}`.

See [config.sae.models.HookPoint](config/sae/models.py) for the available hook points.

Use `HookPoint.ACT.value` to refer to activations between transformer block layers.
For a Transformer with 4 layers, the SAE keys are:
`0_act`, `1_act`, `2_act`, `3_act`, `4_act`
corresponding to the keys:
`0_residpre`, `1_residpre`, `2_residpre`, `3_residpre`, `3_residpost`

For SAEs that come in pairs, the keys are:
`0_residmid`, `0_residpost`, `1_residmid`, `1_residpost`, ... `3_residmid`, `3_residpost`
OR
`0_mlpin`, `0_mlpout`, `1_mlpin`, `1_mlpout`, ... `3_mlpin`, `3_mlpout`


