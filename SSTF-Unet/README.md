# SSTF-Unet: Spatial–Spectral Transformer-Based U-Net for High-Resolution Hyperspectral Image Acquisition
> **Authors:** 
> [*Haibo Liu*](https://scholar.google.com.hk/citations?hl=zh-CN&user=SisjYXYAAAAJ),
> *Chenguo Feng*,
> [*Renwei Dian*](https://scholar.google.com.hk/citations?hl=zh-CN&user=EoTrH5UAAAAJ),
> and [*Shutao Li*](https://scholar.google.com.hk/citations?hl=zh-CN&user=PlBq8n8AAAAJ).
>
> <sup>&#x2709;&#xFE0F;</sup> Corresponding author (drw@hnu.edu.cn).

## Overview
Code implementation of "_**SSTF-Unet: Spatial–Spectral Transformer-Based U-Net for High-Resolution Hyperspectral Image Acquisition**_". IEEE TNNLS 2024. [Paper](https://ieeexplore.ieee.org/document/10260685)

## Environment Setup

- Python 3.9 or higher
- PyTorch 1.12 or higher
- CUDA Toolkit (choose the appropriate version based on your GPU configuration)

## Usage
### 1. Prepare data
We use Dataset.py to process the CAVE and Harvard datasets. You can obtain the training dataset and test dataset according to your own processing method, and then modify the corresponding file paths in Train.py.

### 2. Train
To train SSTF-Unet with costumed path:

```bash
python Train.py
```
### 3. Inference

Select the best performing model and modify the corresponding model weight path in Construct.py. Then, to obtain the reconstructed hyperspectral images:

```bash
python Construct.py
```

## Citation

Please cite our paper if you find the work useful, thanks!

    @article{liu2024sstf,
      title={SSTF-Unet: Spatial–Spectral Transformer-Based U-Net for High-Resolution Hyperspectral Image Acquisition},
      author={Liu, Haibo and Feng, Chenguo and Dian, Renwei and Li, Shutao},
      journal={IEEE Transactions on Neural Networks and Learning Systems},
      year={2024},
      volume={35},
      number={12},
      pages={18222-18236},
      publisher={IEEE}
    }
 
