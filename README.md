
# Watermarking Molecules

Official implementation of [Copyright Protection for 3D Molecular Structures with Watermarking]

## Update
This work is based on the implementation of [Unified Generative Modeling of 3D Molecules with Bayesian Flow Networks] (https://github.com/AlgoMole/GeoBFN) 

## Prerequisite
You will need to have a host machine with gpu, and have a docker with `nvidia-container-runtime` enabled.


## Quick start

### Environment setup
Clone the repo with `git clone`,
```bash
git clone https://github.com/RunwenHU/WMM.git
```

### Train a model on qm9 dataset

python main.py --config_file configs/bfn4molgen.yaml --epochs 3000 --no_wandb








