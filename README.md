# 🦠 Malaria Parasite Detection using Deep Learning & XAI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
**Course:** Data Analytics for Life Science (Final Capstone Project)
**Track:** A - Medical Imaging

This project implements an automated diagnostic tool to detect **Malaria Parasites** in thin blood smear images. It utilizes **Deep Learning (CNN)** to classify cells as *Parasitized* or *Uninfected*.

To ensure clinical trust, the project incorporates **Explainable AI (Grad-CAM)**, allowing users to visualize exactly *where* the model is looking (e.g., highlighting the parasite) before making a decision.

---

## 🚀 Key Features
*   **Two Architectures:**
    *   **Custom CNN (Baseline):** A lightweight, efficient model optimized for edge devices (Accuracy: ~93%).
    *   **VGG16 (Advanced):** A heavy transfer-learning model for maximum recall (Accuracy: ~96%).
*   **Explainable AI (XAI):** Integrated **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate heatmaps overlaying the original image.
*   **Web Application:** A user-friendly **Streamlit** interface for real-time diagnosis.
*   **Data Pipeline:** Automated fetching from TensorFlow Datasets (TFDS) with rigorous preprocessing (Resizing, Normalization, Augmentation).

---

## 📂 Repository Structure

```text
├── app.py                   # The Streamlit Web Application
├── final_notebook.ipynb     # Jupyter Notebook (Data Loading -> Training -> Evaluation)
├── requirements.txt         # List of dependencies
├── README.md                # Project Documentation
├── malaria_model.h5         # The trained model (Generated after running the notebook)
└── .gitignore               # Files to ignore (e.g., large datasets)
