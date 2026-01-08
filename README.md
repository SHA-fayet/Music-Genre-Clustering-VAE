# Hybrid Music Genre Clustering using Beta-VAE

## 📌 Project Overview
This repository implements a **Hybrid Beta-VAE** for unsupervised music clustering, compliant with the Neural Networks Course Project requirements.

## 📂 Repository Structure
* `data/`: Contains audio samples and metadata.
* `src/`: Modular source code (VAE, Clustering, Evaluation).
* `notebooks/`: Exploratory Data Analysis (EDA).
* `results/`: Latent space visualizations and clustering metrics.

## 🚀 Key Features
* **Disentangled Representation:** Implemented using Beta-VAE (beta=4.0).
* **Hybrid Features:** Combines Audio Spectrograms + Text Embeddings.
* **Clustering Analysis:** K-Means, Agglomerative, and DBSCAN comparisons.

## 🛠️ Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the training pipeline (see `notebooks/`).
