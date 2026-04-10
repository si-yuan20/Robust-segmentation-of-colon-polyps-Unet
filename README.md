
---

# 🧠 Enhanced U-Shape Network with Cross-Scale Attention and Frequency-Domain Perception

### for Robust Colorectal Polyp Segmentation

---

## 📌 Overview

This repository corresponds to our research work:

> **An Enhanced U-Shape Architecture Network with Cross-Scale Attention and Frequency-Domain Perception for Robust Colorectal Polyp Segmentation**

Accurate segmentation of colorectal polyps from colonoscopy images plays a critical role in early diagnosis and prevention of colorectal cancer. However, challenges such as scale variation, blurred boundaries, and complex backgrounds significantly hinder segmentation performance.

To address these issues, we propose an enhanced U-shaped encoder–decoder architecture that integrates:

* ✅ **SCAS (Skip Connection-based Cross-scale Attention Strategy)**
* ✅ **D2F (Dual-Domain Feature Fusion with Frequency Perception)**

These modules jointly improve feature representation by capturing both spatial and frequency-domain information.

---

## 🚀 Key Features

* 🔹 Cross-scale attention for effective multi-level feature fusion
* 🔹 Frequency-domain perception for boundary refinement
* 🔹 Robust performance across multiple benchmark datasets
* 🔹 Lightweight and efficient design for practical deployment

---

## 📂 Project Structure

```bash
├── scripts/
│   ├── train.py        # Training pipeline
│   ├── val.py          # Validation and evaluation
│   └── utils.py        # Utility functions (metrics, logging, etc.)
│
├── models/             # Network architecture (SCAS, D2F modules)
├── datasets/           # Dataset loading and preprocessing
├── results/            # Training logs and saved outputs
│   ├── logs/
│   ├── models/
│   └── visuals/
│
├── config/             # Configuration files
├── README.md
```
---

## 📊 Datasets

The model is evaluated on multiple public polyp segmentation datasets:

* **CVC-ClinicDB**
* **CVC-ColonDB**
* **ETIS-LaribPolypDB**
* **CVC-t**

📌 Please organize datasets as:

```bash
datasets/
├── dataset_name/
│   ├── images/
│   └── masks/
```
---

## 📥 Pretrained Models & Resources

All pretrained weights, experimental results, and supplementary materials are available via Google Drive:

👉 **Download Link:**
[https://drive.google.com/drive/folders/1YZtpfXwUjrcIPLoa8odCHRX-lquYAlPy?usp=sharing](https://drive.google.com/drive/folders/1YZtpfXwUjrcIPLoa8odCHRX-lquYAlPy?usp=sharing)

Contents include:

* ✔️ Pretrained model weights
* ✔️ Training logs
* ✔️ Visualization results
* ✔️ Dataset splits (train/val/test)
---

## ⚙️ Installation

```bash
conda create -n polyp python=3.10
conda activate polyp

pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
sh main.sh
```

### Key Settings:

* Batch size: 8
* Learning rate: 1e-4
* Optimizer: AdamW
* Scheduler: Cosine Annealing
* Epochs: 100

---

## 🧪 Evaluation

```bash
python scripts/val.py
```

Evaluation metrics include:

* mDice
* mIoU
* Precision / Recall
* HD95
* ASSD
* BF-score

---

## 📈 Results

Our method demonstrates strong performance across datasets, achieving:

* Superior boundary delineation
* Robust generalization ability
* Stable multi-run performance

---

Saved under:

```bash
results/visuals/
```
---

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@article{your_paper_2026,
  title={Enhanced U-Shape Network with Cross-Scale Attention and Frequency-Domain Perception for Robust Colorectal Polyp Segmentation},
  author={Your Name et al.},
  journal={Plos one},
  year={2026}
}
```
