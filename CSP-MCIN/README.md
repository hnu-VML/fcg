# Correlation and Semantic Prior-Guided Multi-Scale Cross-Modal Interaction Network for SAR-OPT Image Fusion
> **Authors:**
> *Xiaoyang Hou*<sup>&dagger;</sup>,
> *Lingxi Zhou*<sup>&dagger;</sup>,
> *Chenguo Feng*<sup>&dagger;</sup>,
> *Hao Cha*,
> [*Yang Liu*](https://github.com/YonderL),
> *Liguo Liu*<sup>&#x2709;&#xFE0F;</sup>,
> and [*Haibo Liu*](https://scholar.google.com.hk/citations?hl=zh-CN&user=SisjYXYAAAAJ).
>
> <sup>&dagger;</sup> These authors contributed equally to this work.
>
> <sup>&#x2709;&#xFE0F;</sup> Corresponding author (1309021015@nue.edu.cn).


## Overview
Code implementation of "_**Correlation and Semantic Prior-Guided Multi-Scale Cross-Modal Interaction Network for SAR-OPT Image Fusion**_". MDPI Remote Sensing 2026.  [Paper](https://www.mdpi.com/2072-4292/18/7/975)

## Environment Setup

- Python 3.10 or higher
- PyTorch 2.5 or higher
- CUDA Toolkit (choose the appropriate version based on your GPU configuration)

## Dataset Layout

Training, validation, and inference all expect paired SAR and optical images stored in `sar/` and `opt/` subfolders.

```text
dataset_root/
  train/
    sar/
      0001.png
      0002.png
    opt/
      0001.png
      0002.png
  val_or_test/
    sar/
      0003.png
    opt/
      0003.png
```

- Files are paired by the same filename stem. For example, `sar/0001.png` matches `opt/0001.jpg`.
- Supported image formats are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`.
- Keep filename stems unique inside each split because the loader builds pairs by stem name.
- In config files, `data.train_dir` and `data.val_dir` should point to split folders that directly contain `sar/` and `opt/`.

## Configs Used In Our Experiments

The repository keeps several preset configs. Update the dataset paths before running on another machine.

| Config | Dataset | Split setting in config | Notes |
| --- | --- | --- | --- |
| `configs/QS.yaml` | QXSLAB-SAROPT | `train` / `test` | 256x256, single-channel SAR and optical input |
| `configs/SEN1-2.yaml` | SEN1-2 | `train` / `test` | 256x256, single-channel SAR and optical input |
| `configs/WOS.yaml` | WHU-OPT-SAR | `train` / `val` | 256x256, single-channel SAR and optical input |
| `configs/examples.yaml` | Example template | `train` / `val` | Reference config for custom datasets |

## Training

1. Edit one of the YAML files in `configs/`.
2. Set `data.train_dir` and `data.val_dir` to your local dataset paths.
3. Start training with one of the following commands:

```bash
python train.py --config configs/QS.yaml
python train.py --config configs/SEN1-2.yaml
python train.py --config configs/WOS.yaml
```

Optional flags:

- `--runs-dir` changes the base output directory.
- `--resume <run_dir>` resumes from an existing run.

Each run writes to `runs/run_YYYYMMDD_HHMMSS/` with subfolders such as `ckpt/`, `images/`, and `tb/`, plus `config.yaml` and `log.txt`.

Notes:

- Validation checkpoint selection is based on validation loss (`total`) rather than the internal metric suite.
- `logging.eval_save_names` in the YAML config can be used to save selected validation samples into the run directory.
- This public code path keeps only the ResNet-18 `FusionNet_2gate` model and the `pix_semantic_joint_loss_shallow_feature` training loss shared by the provided configs.

## Inference

Use `test.py` on a directory that contains `sar/` and `opt/`:

```bash
python test.py --data-dir /path/to/test_split --checkpoint /path/to/best_epoch.pt
```

By default, results are saved to `<checkpoint>.pt_testresult/`.

Useful flags:

- `--save-rgb` saves RGB fused images by combining fused Y with the optical CbCr channels.
- `--save-intermediates` exports intermediate feature maps when the selected model supports `forward(..., return_features=True)`.

## Pretrained Weights

### Backbone Weights

Place the backbone checkpoints in `./modules` so the corresponding encoders can load them directly.

| Weight file | Expected path | Download |
| --- | --- | --- |
| `RemoteCLIP-RN50.pt` | `modules/RemoteCLIP-RN50.pt` | TBD |
| `SARCLIP-RN50.pt` | `modules/SARCLIP-RN50.pt` | TBD |

### Released Setup

The public code path is shared by the provided dataset configs listed above.

| Dataset | Matching config | Checkpoint | Download |
| --- | --- | --- | --- |
| QXSLAB-SAROPT | `configs/QS.yaml` | TBD | TBD |
| SEN1-2 | `configs/SEN1-2.yaml` | TBD | TBD |
| WHU-OPT-SAR | `configs/WOS.yaml` | Project checkpoint | TBD |

The download links above will be updated after the weight files are uploaded.

## Citation

Please cite our paper if you find the work useful, thanks!

    @article{hou2026correlation,
    title={Correlation and Semantic Prior-Guided Multi-Scale Cross-Modal Interaction Network for SAR-OPT Image Fusion},
    author={Hou, Xiaoyang and Zhou, Lingxi and Feng, Chenguo and Cha, Hao and Liu, Yang and Liu, Liguo and Liu, Haibo},
    journal={Remote Sensing},
    volume={18},
    number={7},
    pages={975},
    year={2026},
    publisher={MDPI}
    }
