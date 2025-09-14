# 🌌🌟 ASTERIS: Pushing Detection Limits of Astronomical Imaging via Self-supervised Spatiotemporal Denoising 

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

**Introduction**: Noise fundamentally constrains the detection limit of astronomical observations. We present **ASTERIS**, a self-supervised denoising transformer **integrating spatiotemporal information across multiple exposures** to enhance detection limits  (the backbone was inspired by **[Restormer](https://github.com/swz30/Restormer)**). Quantitative mock tests are developed for scientific benchmarking, illustrating that ASTERIS improves practical detection limits over existing methods by **1.0 magnitude** at 90% completeness and purity, while preserving point-spread-function fidelity and photometric accuracy. Experimental validations on the 🪐**James Webb Space Telescope (JWST)** and the ☄️**Subaru Telescope** demonstrate ASTERIS’s capability in resolving previously undetectable features.

**Version**: ✨1.0

**Copyright**: 2025, Y. Guo, H. Zhang, M. Li, J. Wu, Z. Cai, and Q. Dai

**doi**: [https://doi.org/****](https://doi.org/****)


---

## 📖 Table of Contents
- [Installation](#-installation)
- [Usage](#-usage)
- [Examples](#-examples)
- [Citation](#-citation)
- [License](#-license)
---

## 🔧 Installation

**Environment**

- Ubuntu 20.04.6
- CUDA 12.1
- CONDA 23.9.0
- Python 3.9.23
- Pytorch 2.5.0+cu121

Other versions may work, but we cannot guarantee full compatibility.

Two ways are recommended to set up the environment. **We recommend using the `environment.yml` method** for reproducibility, but manual installation is also possible.

```bash
# Clone repository
git clone https://github.com/freemercury/ASTERIS_THU.git

# Create environment with environment.yml
conda env create -f environment.yml
```

If you prefer to set up the environment manually, follow these steps:

```bash
# 1. Create a new conda environment with Python 3.9
conda create -n asteris python=3.9

# 2. Activate the environment
conda activate asteris

# 3. Install core dependencies
pip install numpy==1.26
pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# 4. Install any additional packages you need
pip install tqdm scipy astropy tifffile scikit-image einops swanlab natsort pathlib
```


---

## 📂 Usage

ASTERIS is designed for multi-exposure astronomical imaging data denoising. Below is the standard workflow for preparing data, training models, and running tests.


### 1️⃣ Prepare Training Data

- The training process takes as input a set of **3D `.tif` image stacks** under `./train_datasets/`.  
- In each stack, every frame must be **astrometrically aligned to the same WCS reference**, typically generated during the data reduction of astronomical imaging data.
- Pixle scale of images to be denoised by the pre-trained model needs to be set to **0.04 arcseconds**.

### 2️⃣ Convert FITS to Training Stacks

- Make subfolders under `./reduction/`.
- Place each **astronomically aligned `.fits` files** of the same pointing into the corresponding `./reduction/XX/`.
- Each subfolder needs **≥ 8/16 `.fits`** files for **ASTERIS_4/8 training**.
- Run the following script to convert them into `.tif` stacks recognizable by ASTERIS:  

```bash
python ASTERIS_make_train_dataset.py
```

- Script parameters are defined inside `ASTERIS_make_train_dataset.py` and can be tuned as needed.

### 3️⃣ Train a Model

To start training from scratch, run:
```bash
python ASTERIS_train.py
```
- All training parameters are configurable in the script itself.


### 4️⃣ Test the Model

Place your .fits test images under ./test_datasets/.

Ensure:

- All FITS files in the same sub-directory are astrometrically aligned.

- When using ASTERIS_*M*, each sub-directory must contain at least *M* individual `.fits` files — for example, ≥ 4 for ASTERIS4, ≥ 8 for ASTERIS8.

- We provide two variants of pre-trained ASTERIS models--4-frame and 8-frame--that can be directly tested. They are trained separately for JWST/NIRCam long-wavelength (center wavelength ≥ 2.5 μm) and short-wavelength (center wavelength < 2.5 μm), using the imaging data from **Program IDs [3293](https://www.stsci.edu/jwst-program-info/download/jwst/pdf/3293/), [1210](https://www.stsci.edu/jwst/phase2-public/1210.pdf), [3215](https://www.stsci.edu/jwst-program-info/download/jwst/pdf/3215/), and [1963](https://www.stsci.edu/jwst/phase2-public/1963.pdf)**.

- The **[pre-trained model](https://doi.org/10.5281/zenodo.17114980)** can be downloaded here.


Run the testing script:
```bash
python ASTERIS_test.py
```

✅ By following the above steps, you can prepare datasets, train new models, or evaluate ASTERIS on your own astronomical imaging data.

---

## 💡 Examples

We provide demo data and pre-trained model that can be directly tested with ASTERIS.  

1. Download the **[demo data](https://doi.org/10.5281/zenodo.17105027)** from [https://doi.org/10.5281/zenodo.17105027](https://doi.org/10.5281/zenodo.17105027).   
2. Unzip the archive and place the contents into `./test_datasets/`. 
3. Download the **[pre-trained model](https://doi.org/10.5281/zenodo.17114980)** from [https://doi.org/10.5281/zenodo.17114980](https://doi.org/10.5281/zenodo.17114980). 
4. Unzip the archive and place the contents into `./pth/`.  
5. Run the following scripts to test the demo datasets:
```bash
# Test on demo (short wavelengths)
python ASTERIS_test_demo_short.py

# Test on demo (long wavelengths)
python ASTERIS_test_demo_long.py
```
4. The results will be automatically saved under `./results/`.

---

## 📝 Citation

If you use this code, please cite the companion paper where the original method appeared:

Guo Y., Zhang H., Li M., et al. "Pushing Detection Limits of Astronomical Imaging via Self-supervised Spatiotemporal Denoising". Science (2025). doi: [https://doi.org/****](https://doi.org/****)

<pre>@article{Guo2025ASTERIS, 
title = {Pushing Detection Limits of Astronomical Imaging via Self-supervised Spatiotemporal Denoising}, 
author = {Guo, Yuduo and Zhang, Hao and Li, Mingyu and Yu, Fujiang and Wu, Yunjing and  Hao, Yuhan and Huang, Song and Liang, Yongming and Lin, Xiaojing and Li, Xinyang and Wu, Jiamin and Cai, Zheng and Dai, Qionghai}, 
journal = {Science}, 
year = {2025}, 
volume = {999}, 
pages = {1--12}, 
doi = {10.1234/astro.2025.12345}
}</pre>

Should you have any questions regarding this code and the corresponding results, please contact Yuduo Guo (gyd@mail.tsinghua.edu.cn).

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
