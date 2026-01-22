下面是一份**可直接用于 GitHub / GitLab 项目的 README（英文版，符合计算机视觉/医学影像论文代码仓库常见规范）**，内容与你给出的论文摘要、方法和实验结果严格一致，并**明确注明“论文录用后公开代码与权重”**，语气学术、克制、规范。

---

# Enhanced U-Shape Network with Cross-Scale Attention and Frequency-Domain Perception

for Robust Colorectal Polyp Segmentation

## 📌 Overview

This repository corresponds to the paper:

> **An Enhanced U-Shape Architecture Network with Cross-Scale Attention and Frequency-Domain Perception for Robust Colorectal Polyp Segmentation**

Early detection and accurate segmentation of colorectal polyps in colonoscopy images are crucial for reducing the incidence and mortality of colorectal cancer. However, polyp segmentation remains challenging due to large-scale variations in polyp size, indistinct boundaries, and complex background interference.

To address these challenges, we propose an enhanced U-shaped encoder–decoder network that integrates **cross-scale attention mechanisms** and **frequency-domain perception**, enabling robust and accurate colorectal polyp segmentation.

⚠️ **Note**:
**The source code, pretrained model weights, and full experimental configurations will be publicly released after the acceptance of the corresponding paper.**

---

## 🧠 Methodology

The proposed network introduces two key modules to enhance feature representation and boundary delineation:

### 1. Scale-Aware Cross Attention Skip Connection (SCAS)

* Designed for encoder–decoder skip connections
* Utilizes **multi-head cross attention** to model interactions between encoder and decoder features
* Integrates **multi-scale pyramid fusion** to dynamically adapt to polyps of varying sizes
* Effectively alleviates semantic gaps between low-level and high-level features

### 2. Dual-Domain Fusion (D2F) Module

* Deployed at the bottleneck stage of the network
* Separates features into:

  * **Low-frequency components** (global structural information)
  * **High-frequency components** (edge and boundary details)
* Assigns adaptive weights to different frequency components
* Enhances joint spatial–frequency domain perception for precise boundary segmentation

---

## 🏗️ Network Architecture

* U-shaped encoder–decoder backbone
* Cross-scale attention-based skip connections
* Frequency-domain perception at the latent representation stage
* End-to-end trainable with standard segmentation losses

---

## 📊 Experimental Results

The proposed method is evaluated on multiple public colorectal polyp segmentation benchmarks.

### Quantitative Performance (Representative Results)

| Dataset      | mIoU   | mDice  |
| ------------ | ------ | ------ |
| CVC-ClinicDB | 0.8738 | 0.9314 |

The model demonstrates **competitive or superior performance** compared with existing mainstream approaches, particularly in handling:

* Multi-scale polyp morphology
* Blurred or irregular boundaries
* Background noise and visual artifacts

---

## 📁 Repository Structure (Planned)

```text
.
├── configs/            # Training and evaluation configurations
├── datasets/           # Dataset loading and preprocessing scripts
├── models/
│   ├── backbone/       # Encoder–decoder backbone
│   ├── scas/           # Scale-Aware Cross Attention module
│   ├── d2f/            # Dual-Domain Fusion module
│   └── network.py      # Full network definition
├── losses/             # Loss functions
├── utils/              # Metrics, logging, visualization
├── train.py            # Training script
├── test.py             # Evaluation script
└── README.md
```

---

## 📦 Code & Model Availability

🚧 **Current Status**:
The codebase and pretrained model weights are **under preparation**.

📢 **Release Plan**:

* Full source code
* Pretrained model weights
* Training and evaluation scripts
* Detailed reproduction instructions

👉 **All materials will be publicly released immediately after the paper is officially accepted.**

---

## 🏥 Clinical Significance

The proposed method provides a reliable and accurate segmentation tool for colorectal polyps, offering potential support for **clinical endoscopic assistant diagnosis**, and contributing to improved colorectal cancer screening workflows.

---

## 📄 Citation

If you find this work useful in your research, please consider citing our paper (citation information will be updated upon publication).

---

## 📬 Contact

For questions or collaboration inquiries, please contact the authors via the correspondence information provided in the paper.

---
