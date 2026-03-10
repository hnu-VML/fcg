# SSSMN: Spatial-Spectral Sparse Mamba Network for Efficient Hyperspectral Fusion Super-resolution
> **Authors:**
> *Chenguo Feng*,
> [*Haibo Liu*](https://scholar.google.com.hk/citations?hl=zh-CN&user=SisjYXYAAAAJ),
> [*Renwei Dian*](https://scholar.google.com.hk/citations?hl=zh-CN&user=EoTrH5UAAAAJ),
> [*Yang Liu*](https://github.com/YonderL),
> and [*Shutao Li*](https://scholar.google.com.hk/citations?hl=zh-CN&user=PlBq8n8AAAAJ).

## Overview
Code implementation of "_**SSSMN: Spatial-Spectral Sparse Mamba Network for Efficient Hyperspectral Fusion Super-resolution**_". ELSEVIER PR 2026.

## Environment Setup

- Python 3.8 or higher
- PyTorch 1.13 or higher
- CUDA Toolkit (choose the appropriate version based on your GPU configuration)

## Citation

Please cite our paper if you find the work useful, thanks!

    @article{feng2026sssmn,
      title={SSSMN: Spatial-Spectral Sparse Mamba Network for Efficient Hyperspectral Fusion Super-resolution},
      author={Feng, Chenguo and Liu, Haibo and Dian, Renwei and Liu, Yang and Li, Shutao},
      journal={Pattern Recognition},
      year={2026},
      publisher={Elsevier}
    }

<!--
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
-->
