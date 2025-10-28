
# MolMark: Safeguarding Molecular Structures with Atom-Level Watermarking

Official implementation of [MolMark: Safeguarding Molecular Structures with Atom-Level Watermarking]

## Update
This work is based on the implementation of [Unified Generative Modeling of 3D Molecules with Bayesian Flow Networks] (https://github.com/AlgoMole/GeoBFN) 
## Related works
(1) FoldMark: Safeguarding Protein Structure Generative Models with Distributional and Evolutionary Watermarking (https://github.com/zaixizhang/FoldMark)

(2) Securing the Language of Life: Inheritable Watermarks from DNA Language Models to Proteins

## Prerequisite
You will need to have a host machine with gpu, and have a docker with `nvidia-container-runtime` enabled.

## Overview
MolMark is the first watermarking strategy designed to protect moleucles. It:
- **Maintains Molecular Properties:** Operates at the atom level, embedding watermarks by subtly modulating chemically informed features.
- **Ensure Molecular Functionlaity:** Guarantes the physicochemical properties and the functionality in docking performance.
- **Exhibits High Bit Accuracy:** Achieves over 95% watermark bit accuracy at 16 bits with minimal impact on structural integrity.
- **Presents Robust Against SE(3) Transformations:** Has high robustness against rotation, translation, and reflection with bit accuracy higher than 90%.


## Results
### Application scenarios and structures of MolMark in protecting molecules
<div align=center>
<img src="https://github.com/RunwenHu/MolMark/blob/main/results/fig1.jpg" width="600"/>
</div>

### Structure of eight pairs of molecules
<div align=center>
<img src="https://github.com/RunwenHu/MolMark/blob/main/results/fig2.jpg" width="600"/>
</div>


## Quick start

### Environment setup
Clone the repo with `git clone`,
```bash
git clone https://github.com/RunwenHU/MolMark.git
```

### Train a model on qm9 dataset

python main.py --config_file configs/bfn4molgen.yaml --epochs 3000 --no_wandb


## Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{hu2024molmark,
  title={MolMark: Safeguarding Molecular Structures with Atom-Level Watermarking},
  author={Hu, Runen and Chen, Peilin and Ding, Keyan and Wang, Shiqi},
  journal={arXiv},
  year={2025},
}
```





