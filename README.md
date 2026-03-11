# Pathology-Aware Latent Diffusion for Low-Field Brain MRI Enhancement
[![Paper](https://img.shields.io/badge/Paper-TMI-blue)]()
[![Python](https://img.shields.io/badge/Python-3.9-green)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()

Official implementation of the paper:

**Pathology-Aware Latent Diffusion for Low-Field Brain MRI Enhancement**  
Dong Zhang, Caohui Duan, Xiaonan Xu, Youmin Li, Junming Huang, Chao Wang, Z. Jane Wang, Xin Lou  
IEEE Transactions on Medical Imaging (TMI)

---

## Overview

Low-field MRI (LF-MRI) improves imaging accessibility and reduces hardware cost, but suffers from reduced signal-to-noise ratio (SNR) and degraded anatomical detail.

Most existing enhancement approaches rely solely on image supervision and may fail to preserve pathology-relevant structures.

We propose **RePaMed**, a **pathology-aware latent diffusion framework** that integrates **diagnostic report semantics** to guide MRI enhancement.

The key idea is to align **region-level MRI representations with diagnostic report semantics**, and incorporate the aligned representations into a **diffusion-based reconstruction model**.

---

## Framework

<p align="center">
<img src="docs/fig2.png" width="85%">
</p>




# Supplementary Material  

# S1 Data Preparation Details

## S1.1 Data Acquisition

### Table S1. Dataset characteristics across sites

| | Training & Validation | | | Testing | |
|---|---|---|---|---|---|
| Hospital | PLA General | Guizhou Electrical | Raohe County | PLA General | Beian Central |
| No. MRIs | 80,222 | 1,230 | 485 | 1,392 | 54 |
| No. patients | 68,754 | 922 | 485 | 1,392 | 54 |
| Field strength (T) | 3.0 | 0.4 | 0.35 | 3.0 / 0.3–0.4 | 3.0 / 0.3 |
| Time | 2018–2023 | 2021–2024 | 2018–2024 | 2023 | 2025 |
| Patient age (y) | 56.9 ± 18 | 55.4 ± 14 | N/A | 58.3 ± 15 | 59.5 ± 6 |
| Male (%) | 49.5 | 62.9 | 48.1 | 51.2 | 64.8 |
| MRI scanner | GE, Philips, Siemens | Hitachi | Time Medical | GE, Siemens | Siemens / Hitachi |

**Pixel spacing (mm)**  

| PLA | Guizhou | Raohe | PLA test | Beian |
|---|---|---|---|---|
| 0.47×0.47×7.0 | 0.86×0.86×7.0 | 0.57×0.57×8.5 | 0.47×0.47×7.0 / 0.94×0.94×7.0 | 0.43×0.43×7.15 / 0.86×0.86×7.15 |

**Dimensions**

| PLA | Guizhou | Raohe | PLA test | Beian |
|---|---|---|---|---|
| 512×512×21 | 256×256×20 | 420×420×15 | 512×512×21 / 256×256×21 | 512×512×19 / 256×256×19 |

Training and validation data were collected from one tertiary hospital and two secondary hospitals. HF MRI data were acquired at the Chinese PLA General Hospital using multiple scanners.

Real LF MRI data (0.35–0.4T) were collected from two secondary hospitals and used:

- to calibrate simulation parameters  
- to train unpaired enhancement baselines

Independent testing data were collected from PLA General Hospital and Beian Central Hospital.

Since PLA provides only 3T MRI, paired LF data were simulated using the degradation pipeline. Beian Central Hospital provides real paired HF/LF acquisitions for cross-domain evaluation.

---

# S1.2 MRI Pre-processing

All MRIs were preprocessed using the following pipeline:

1. **Anonymization** to remove patient identifiers  
2. **Skull stripping** to isolate intracranial tissue  
3. **N4 bias field correction** to reduce intensity inhomogeneity  
4. **Cropping** to fixed spatial size  

HF: `[448 × 448 × 18]`  
LF: `[224 × 224 × 18]`

The same pipeline was applied to training and test datasets.

---

# S1.3 Radiology Report Filtering and Cleaning

Reports were filtered to remove invalid cases containing keywords such as:

- “preoperative localization”
- severe motion blur descriptions

Remaining reports were cleaned using a locally deployed **Qwen3-235B** model to:

- remove extracranial content
- remove redundant normal findings
- preserve abnormal locations and diagnostic conclusions

All cleaned reports were manually verified by clinicians.

---

# S1.4 Physics-Inspired LF MRI Degradation

Realistic LF MRI was simulated from HF MRI using a physics-inspired pipeline.

## Tissue-aware SNR modeling

HF MRIs were segmented into four tissue classes using **FreeSurfer**:

- White Matter (WM)
- Gray Matter (GM)
- CSF
- Others

Empirically measured SNR ranges:

| Tissue | SNR |
|---|---|
| WM | 12–15 |
| GM | 10–12 |
| CSF | 5–6 |
| Others | 8–9 |

Noise was injected proportional to signal intensity.

---

## Resolution degradation

Images were resampled to [224 × 224 × 18] 
An anisotropic Gaussian kernel (σ=0.5–1.3) simulated slice-profile blurring.

---

## Bias-field simulation

Multiplicative bias fields were generated using polynomial functions (degree 2–4).

Coefficient range:
0.08 – 0.28


---

## Artifact injection

Low-probability artifacts were added:

- ghosting
- RF spikes
- bias field shifts

Final LF images were normalized to `[0,1]`.

---

# S1.5 Anatomical Parser Training

FreeSurfer labels were mapped into **25 anatomical structures**.

Using **2,560 labeled MRIs**, a **SwinUNETR** segmentation network was trained.

The trained model is used for:

- region-wise evaluation
- anatomical consistency analysis

---

# S2 Implementation Details

## S2.1 RePaAlign — Report–Pathology Alignment

### Visual encoder

Architecture:
ResNet-18 + Feature Pyramid Network + Convolutional Attention Module


Slice features are aggregated into structure-wise embeddings using mask-weighted pooling.

Each embedding is concatenated with its anatomical name embedding obtained from **clinical BERT**.

---

### Report encoder and contrastive learning

Reports are encoded using **medbert-base-wwm-chinese**.

Contrastive learning uses structured hard negatives:

- anatomical entity remapping
- sentence shuffling
- cross-patient substitution
- hybrid mixing

Training objective: bidirectional image–text contrastive loss


After training:

- report encoder removed
- visual encoder frozen
- reused by RePaDiff

---

# S2.2 RePaDiff — Pathology-Aware Diffusion

## Latent diffusion backbone

Based on **Latent Diffusion Model (LDM)**.

A VAE is pretrained on HF MRI.

Latent size: [8 x 56 x 56]


---

## Conditioning signals

Two conditioning inputs:

**1. Structural conditioning**

LF encoder projects LF MRI into latent space.

**2. Semantic conditioning**

RePaAlign visual encoder extracts pathology embeddings.

Embedding size: [1 x 128]


---

## Curriculum Conditioning Injection (CCI)

Semantic embeddings are injected through cross-attention.

A time-dependent coefficient gradually increases during diffusion steps:

early → structure reconstruction
late → pathology refinement


---

## Optimization

Training objective: denoising score matching


Trainable modules:

- LF encoder
- conditional UNet

Frozen modules:

- VAE
- visual encoder

---

## Training configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| GPUs | 4 × NVIDIA A100-80GB |
| RePaAlign optimizer | Adam |
| RePaDiff optimizer | AdamW |
| Learning rate | 1e-4 |
| Diffusion steps (train) | 1000 |
| Diffusion steps (test) | 50 |

Training can also run on **RTX 4090** using gradient accumulation.

---

# S3 Complete Results

## Quantitative results

Table S2 reports pixel-level metrics.

Metrics:

- MSE
- MAE
- SSIM
- PSNR
- LPIPS
- HFEN
- GMSD

The proposed method achieves the best trade-off between structural fidelity and perceptual quality.

---

## Structural consistency

Table S3 reports **Relative Volume Error (RVE)** for six brain tissues:

- White matter
- Cortical gray matter
- Subcortical nuclei
- Cerebellum
- Brainstem
- Ventricle

Lower RVE indicates better anatomical consistency.

---

# Visual Comparisons

### Fig S1 — Simulated LF MRI

<p align="center">
<img src="doc/result_1_2.png" width="900">
</p>

Comparison of LF→HF enhancement methods on simulated LF MRI.

The proposed method preserves anatomical boundaries and suppresses noise.

---

### Fig S2 — Real LF MRI

<p align="center">
<img src="doc/result_2_2.png" width="900">
</p>

Results on real LF MRI from secondary hospitals.

The proposed method generates clearer tissue contrast and fewer artifacts.

---

### Fig S3 — Segmentation overlays

<p align="center">
<img src="doc/result_3_2.png" width="700">
</p>

Segmentation contours for ventricles and brainstem.

---

### Fig S4 — Error maps and attention

<p align="center">
<img src="doc/result_4_2.png" width="600">
</p>

The proposed method produces more localized reconstruction errors and anatomically meaningful attention.

---
