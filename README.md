# spatial-cell-composition
Submission for Global AI Hackathon'25 by Elucidata

# Predicting Spatial Cell-Type Composition from Histology Images

This project tackles a real-world biomedical machine learning problem: predicting the spatial abundance of 35 cell types from high-resolution H&E-stained histology images using deep learning.

## 🧠 Motivation

Spatial transcriptomics is powerful but expensive. Histology is cheap but coarse. The challenge is to bridge the two with AI — mapping visual signals to molecular insights.

## 🧪 Dataset

- Provided via Kaggle (Elucidata Global AI Hackathon 2025)
- 6 training slides, 1 test slide
- For each slide:
  - HE image (float32 RGB)
  - ~2000 spatial transcriptomic spots
  - Each spot has abundance data for 35 cell types (C1–C35)

## 🔍 EDA Highlights

- Visualized tissue coverage and spot distribution
- Mapped cell-type heatmaps over tissue
- Found biologically meaningful spatial correlations
- Previewed high-abundance patches per cell type

## 🧠 Model

- ResNet18 pretrained on ImageNet
- Modified for 35-channel regression output
- Trained with MSE loss (can upgrade to Spearman rank loss)
- Extracted 224x224 patches per spot

## 🔄 Pipeline

1. Load `.h5` data and extract patches
2. Train CNN on spot-wise cell composition
3. Predict on test slide S_7
4. Export `submission.csv` for Kaggle

## 📊 Results

- Baseline model achieved **%**
- Strongest correlations with structural cell types
- Future work: try EfficientNet, ViT, and integrate multimodal embeddings

## 🛠 Tech Stack

- Python, PyTorch, torchvision, h5py, pandas, matplotlib
- Jupyter Notebooks for reproducibility

## 📸 Sample Outputs

Include screenshots:
- Spot overlay on slide
- Heatmaps of cell type abundance
- Correlation matrix
- Top-k patch previews

## 📁 Submission

Includes a valid `submission.csv` for Kaggle.

---

Want to see the code? Check out:
- [`notebook/`](notebooks/) — fully interactive Jupyter flows
- [`src/`](src/) — clean reusable modules
