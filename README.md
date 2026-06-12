# ASTERIS: Deeper Detection Limits in Astronomical Imaging Using Self-supervised Spatiotemporal Denoising

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper (Science)](https://img.shields.io/badge/Science-doi%3A10.1126%2Fscience.ady9404-red.svg)](https://www.science.org/doi/10.1126/science.ady9404)
[![arXiv](https://img.shields.io/badge/arXiv-2602.17205-b31b1b.svg)](https://arxiv.org/abs/2602.17205)
[![Model](https://img.shields.io/badge/model-Zenodo-orange.svg)](https://doi.org/10.5281/zenodo.17114980)
[![Demo Data](https://img.shields.io/badge/demo_data-Zenodo-orange.svg)](https://doi.org/10.5281/zenodo.17105027)
[![Homepage](https://img.shields.io/badge/Author-Yuduo_Guo-blueviolet.svg)](https://freemercury.github.io)

**ASTERIS** is a self-supervised denoising framework for multi-exposure astronomical imaging. It integrates spatiotemporal information across exposures using a modified [Restormer](https://github.com/swz30/Restormer) backbone, improving practical detection limits by **1.0 magnitude** at 90% completeness and purity while preserving PSF fidelity and photometric accuracy.

Validated on the **James Webb Space Telescope (JWST)** and the **Subaru Telescope**.

> Y. Guo, H. Zhang, M. Li, F. Yu, Y. Wu, Y. Hao, S. Huang, Y. Liang, X. Lin, X. Li, J. Wu, Z. Cai, Q. Dai. *Deeper detection limits in astronomical imaging using self-supervised spatiotemporal denoising*. **Science** (2025). [doi:10.1126/science.ady9404](https://www.science.org/doi/10.1126/science.ady9404)

---

## Pre-trained Models

| Model | Wavelength | Frames | Download |
|-------|-----------|--------|----------|
| `ASTERIS8_nrcshort` | < 2.5 um | 8 | [Zenodo](https://doi.org/10.5281/zenodo.17114980) |
| `ASTERIS8_nrclong` | >= 2.5 um | 8 | [Zenodo](https://doi.org/10.5281/zenodo.17114980) |
| `ASTERIS4_nrcshort` | < 2.5 um | 4 | [Zenodo](https://doi.org/10.5281/zenodo.17114980) |
| `ASTERIS4_nrclong` | >= 2.5 um | 4 | [Zenodo](https://doi.org/10.5281/zenodo.17114980) |

These models can be directly applied to **JWST/NIRCam data from F070W to F480M** with arbitrary exposure time. Pixel scale must be **0.04 arcsec/pixel**.

---

## Installation

**Requirements**: Ubuntu 20.04, CUDA 12.1, Python 3.9, PyTorch 2.5.0

```bash
git clone https://github.com/freemercury/ASTERIS_THU.git
cd ASTERIS_THU

# Option 1: from environment file (recommended)
conda env create -f environment.yml

# Option 2: manual setup
conda create -n asteris python=3.9
conda activate asteris
pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26 tqdm scipy astropy tifffile scikit-image einops swanlab natsort
```

---

## Quick Start: Demo

```bash
# 1. Download demo data -> ./test_datasets/
#    https://doi.org/10.5281/zenodo.17105027

# 2. Download pre-trained models -> ./pth/
#    https://doi.org/10.5281/zenodo.17114980

# 3. Run
python ASTERIS_test_demo_short.py   # short wavelengths (< 2.5 um)
python ASTERIS_test_demo_long.py    # long wavelengths (>= 2.5 um)

# Results saved to ./result/
```

---

## Usage

### Inference on Your Data

1. Place astrometrically-aligned `.fits` files in subdirectories under `./test_datasets/`
   - ASTERIS_8 requires >= 8 files per subdirectory
   - ASTERIS_4 requires >= 4 files per subdirectory

2. Configure `ASTERIS_test.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `denoise_model` | Model name (e.g., `ASTERIS8_nrclong`) | - |
| `test_mode` | Number of input frames: `4` or `8` | `8` |
| `GPU` | GPU index (e.g., `'0'`, `'0,1'`) | `'0'` |
| `patch_xy` | Patch size (must be divisible by 8) | `800` |
| `overlap_factor` | Overlap between patches | `0.1` |
| `hdu_num` | FITS HDU index for science data | `1` |

3. Run:
```bash
python ASTERIS_test.py
```

### Training a Custom Model

1. Place astrometrically-aligned `.fits` files under `./reduction_datasets/<pointing>/`
   - Each subdirectory needs >= 8 files (for ASTERIS_4) or >= 8 files (for ASTERIS_4)

2. Convert to training stacks:
```bash
python ASTERIS_make_train_dataset.py
```

3. Train:
```bash
python ASTERIS_train.py
```

Key training parameters (set inside the script):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `train_mode` | Frame count: `4` or `8` | `8` |
| `GPU` | GPU indices | `'0,1,2,3,4,6,7'` |
| `n_epochs` | Training epochs | `20` |
| `learning_rate` | Learning rate | `1.5e-4` |
| `patch_xy` | Patch size | `128` |
| `batch_size` | Per-GPU batch size | `3` |

---

## Project Structure

```
ASTERIS_THU/
├── ASTERIS_train.py                 # Training entry point
├── ASTERIS_test.py                  # Inference entry point
├── ASTERIS_test_demo_short.py       # Demo: short wavelengths
├── ASTERIS_test_demo_long.py        # Demo: long wavelengths
├── ASTERIS_make_train_dataset.py    # FITS -> training stacks
├── asteris/
│   ├── ASTERIS_net_4.py             # 4-frame network
│   ├── ASTERIS_net_8.py             # 8-frame network
│   ├── train.py                     # Training logic
│   ├── test.py                      # Inference logic
│   ├── data_process.py              # Data loading and patching
│   └── utils.py                     # FITS I/O, stacking, preprocessing
├── pth/                             # Pre-trained model weights
├── train_datasets/                  # Training data (3D .tif stacks)
├── test_datasets/                   # Test data (.fits files)
└── result/                          # Inference outputs
```

---

## Citation

```bibtex
@article{Guo2025ASTERIS,
  title   = {Deeper detection limits in astronomical imaging using
             self-supervised spatiotemporal denoising},
  author  = {Guo, Yuduo and Zhang, Hao and Li, Mingyu and Yu, Fujiang
             and Wu, Yunjing and Hao, Yuhan and Huang, Song
             and Liang, Yongming and Lin, Xiaojing and Li, Xinyang
             and Wu, Jiamin and Cai, Zheng and Dai, Qionghai},
  journal = {Science},
  year    = {2025},
  doi     = {10.1126/science.ady9404}
}
```

---

## Contact

For questions or collaboration inquiries: **[Yuduo Guo](https://freemercury.github.io)** - gyd@mail.tsinghua.edu.cn

## License

MIT License. Copyright (c) 2025 Yuduo Guo. See [LICENSE](LICENSE) for details.
