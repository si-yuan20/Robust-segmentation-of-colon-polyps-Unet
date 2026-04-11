
# 🧠 Enhanced U-Shape Network with Cross-Scale Attention and Frequency-Domain Perception

### for Robust Colorectal Polyp Segmentation

---

## 📌 Overview

This repository corresponds to our research work:

> **An Enhanced U-Shape Architecture Network with Cross-Scale Attention and Frequency-Domain Perception for Robust Colorectal Polyp Segmentation**

Accurate segmentation of colorectal polyps from colonoscopy images plays a crucial role in early diagnosis and prevention of colorectal cancer. However, challenges such as large-scale variation, indistinct boundaries, and complex background interference remain significant.

To address these issues, we propose an enhanced U-shaped encoder–decoder architecture integrating:

* **SCAS (Skip Connection-based Cross-scale Attention Strategy)**
* **D2F (Dual-domain Feature Fusion with Frequency Perception)**

These modules collaboratively enhance both global structural representation and fine-grained boundary modeling.

---

## 🚀 Key Features

* Cross-scale attention for multi-level feature interaction
* Frequency-domain perception for boundary enhancement
* Robust generalization across multiple datasets
---

## 📂 Project Structure

```bash
├── scripts/
│   ├── train.py        # Training pipeline
│   ├── val.py          # Validation and evaluation
│   ├── test.py         # Testing script
│   ├── model.py        # Network definition
│   ├── config.py       # Configuration settings
│   ├── ImageLoader2D.py# Dataset loader
│   └── utils.py        # Utilities (metrics, logging, etc.)
│
├── datasets/           # Dataset organization
├── checkpoints/        # Saved model weights
├── logs/               # Training logs
├── visual/             # Visualization results
├── split_indices/      # Dataset split files
│
├── main.py
├── main.sh
├── requirement.txt
└── README.md
```

---

## 📊 Datasets

The model is evaluated on widely used polyp segmentation benchmarks:

* CVC-ClinicDB
* CVC-ColonDB
* ETIS-LaribPolypDB
* CVC-T

Dataset structure:

```bash
datasets/
├── dataset_name/
│   ├── images/
│   └── masks/
```

---

## 📥 Pretrained Models & Resources

All pretrained weights, dataset splits, logs, and visualization results are provided via Baidu Netdisk:

👉 **Baidu Netdisk Link:**
[(https://pan.baidu.com/s/1XY--556TUJpaaocnlWlYQQ)](https://pan.baidu.com/s/1XY--556TUJpaaocnlWlYQQ))

🔑 **Extraction Code:** `2ifu`

📦 Contents include:

* Pretrained model weights
* Image datasets
* Dataset split indices (train/val/test)

---

## ⚙️ Installation

```bash
conda create -n polyp python=3.10
conda activate polyp

pip install -r requirement.txt
```
---

## 🏋️ Training

```bash
python scripts/train.py
```

### Training Settings

* Batch size: 8
* Learning rate: 1e-3
* Optimizer: AdamW
* Scheduler: Cosine Annealing
* Epochs: 200

---

## 🧪 Evaluation

```bash
python scripts/val.py
```

### Metrics

* mDice
* mIoU
* Precision / Recall
* HD95
* ASSD
* BF-score

## 📌 Visualization

Segmentation results and comparisons are saved in:

```bash
visual/
```


## 📜 Citation

```bibtex
@article{paper_2026,
  title={Enhanced U-Shape Network with Cross-Scale Attention and Frequency-Domain Perception for Robust Colorectal Polyp Segmentation},
  author={Rong Gao, Qi Ke, Aiquan Li, Xinning Qin, Sichao Zhao*},
  journal={Plos one },
  year={2026}
}
```
