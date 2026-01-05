# 🧠 AI vs Real Image Detection

A deep learning project that detects whether an image is **AI-generated or real** using a **hybrid ensemble approach** combining a **custom-built CNN** and a **fine-tuned Vision Transformer (ViT)**.

This project focuses on real-world applicability in **AI content detection, digital forensics, and media verification**.

---

## 🚀 Project Overview

With the rapid growth of generative AI tools, distinguishing AI-generated images from real ones has become a critical challenge.

This project solves that problem using:
- A **Convolutional Neural Network (CNN) built completely from scratch**
- A **Vision Transformer (ViT) fine-tuned from a powerful pre-trained foundation model (Demigod ViT)**

An **ensemble strategy** combines both models to produce a more accurate and robust final prediction.

---

## 🧩 Model Architecture

### 🔹 Custom CNN (Built From Scratch)
- Implemented manually using **PyTorch**
- Designed to learn low-level visual features such as:
  - Edges
  - Textures
  - AI-generation artifacts
- Trained specifically for **binary classification (AI vs Real)**

### 🔹 Vision Transformer (ViT – Fine-Tuned)
- Based on a **pre-trained ViT foundation model**
- Fine-tuned on AI-generated and real image datasets
- Captures **global context and semantic patterns** that CNNs may miss

### 🔹 Ensemble Strategy
- Combines predictions from CNN and ViT
- Uses probability-based decision logic
- Improves generalization and reduces false predictions

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Hugging Face Transformers  
- Torchvision  
- OpenCV  
- PIL  
- Vision Transformer (ViT)

---

## 📌 Key Features

- ✅ CNN built completely from scratch  
- ✅ ViT fine-tuned from a foundation model  
- ✅ Ensemble-based prediction system  
- ✅ Image-level inference support  
- ✅ Extendable to video frame analysis  
- ✅ Production-ready inference pipeline  

---

## 🎯 Use Cases

- AI-generated image detection  
- Fake media & deepfake analysis  
- Content moderation  
- Digital forensics  
- Computer vision research  

---

## 🔗 Pretrained Model (Hugging Face)

If you want to **try or integrate the fine-tuned ViT model**, it is available on Hugging Face:

👉 **Hugging Face Model:**  
https://huggingface.co/Abin90p/vit-ai-vs-real

---

## 👤 Author

**Abin**  
Aspiring Full-Stack AI Engineer  
Deep Learning | Computer Vision | AI Systems
