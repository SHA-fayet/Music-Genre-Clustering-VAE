


# 🎵 Hybrid Music Genre Clustering via Beta-VAE

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Clustering](https://img.shields.io/badge/Unsupervised_Learning-F7931E?style=flat-square)

> An unsupervised learning approach to music genre classification utilizing disentangled latent representations. Developed as a comprehensive implementation for advanced Neural Networks coursework.

##  Project Overview

This repository provides a complete pipeline for clustering music genres without labeled data. By implementing a **Hybrid Beta-VAE** architecture, the model learns complex, disentangled representations of audio by synthesizing both acoustic characteristics and textual metadata, allowing for highly accurate downstream clustering.

##  Key Features

* **Disentangled Latent Space:** Implemented a Beta Variational Autoencoder ($\beta=4.0$) to enforce statistical independence among learned features, improving interpretability.
* **Hybrid Feature Extraction:** Fuses high-dimensional Audio Spectrograms with contextual Text Embeddings to capture a richer representation of musical data.
* **Comparative Clustering Analysis:** Evaluates the latent space using multiple unsupervised algorithms, including **K-Means**, **Agglomerative Clustering**, and **DBSCAN**, complete with performance metrics.

## 📂 Repository Structure


├── data/           # Raw audio samples, extracted spectrograms, and textual metadata
├── notebooks/      # Jupyter notebooks for Exploratory Data Analysis (EDA) and pipeline testing
├── results/        # Exported latent space visualizations (t-SNE/PCA) and clustering metric reports
├── src/            # Modular source code
│   ├── model/      # Beta-VAE architecture and loss functions
│   ├── cluster/    # K-Means, Agglomerative, and DBSCAN implementations
│   └── evaluate/   # Metric calculations and plotting utilities
└── requirements.txt


## 🛠️ Installation & Usage

**1. Clone the repository**


git clone [https://github.com/SHA-fayet/Music-Genre-Clustering-VAE.git](https://github.com/SHA-fayet/Music-Genre-Clustering-VAE.git)
cd Music-Genre-Clustering-VAE


**2. Install dependencies**
Ensure you have Python 3.8+ installed, then run:


pip install -r requirements.txt



**3. Run the pipeline**
Navigate to the `notebooks/` directory to execute the end-to-end training and evaluation pipeline, or utilize the modular scripts in `src/` for custom training configurations.


jupyter notebook
