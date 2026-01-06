# 🧠 AI vs Real Image Detection (Vision Transformer)

A deep learning project that fine-tunes a **Vision Transformer (ViT)** model to classify images as **REAL** or **AI-GENERATED (FAKE)**. This project uses **PyTorch** and **Hugging Face Transformers**, and is designed to be **interview-ready, modular, and production-friendly**.

---

## 🚀 Project Overview

With the rapid growth of generative AI, distinguishing real images from AI-generated ones has become critical. This project addresses that problem by:

* Leveraging a **pretrained Vision Transformer (ViT)** model
* Freezing the ViT backbone to retain learned visual features
* Fine-tuning only the classification head for binary classification
* Using Hugging Face's **Trainer API** for clean and scalable training

---

## 🧩 Model Architecture

* **Backbone:** Vision Transformer (ViT)
* **Input Size:** 224 × 224 RGB images
* **Output Classes:**

  * `0 → REAL`
  * `1 → FAKE`

### 🔒 Transfer Learning Strategy

* All ViT backbone layers are **frozen**
* Only the **classification head** is trained
* This reduces overfitting and speeds up training

---

## 📂 Dataset Structure

The dataset must follow this folder structure:

```
dataset/
├── train/
│   ├── REAL/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── FAKE/
│       ├── img1.jpg
│       └── img2.jpg
│
└── test/
    ├── REAL/
    └── FAKE/
```

---

## 🛠️ Tech Stack

* **Python 3.9+**
* **PyTorch**
* **Hugging Face Transformers**
* **Scikit-learn**
* **Pillow (PIL)**

---

## ⚙️ Installation

```bash
pip install torch transformers scikit-learn pillow
```

(Optional, for GPU support)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🧪 Training Configuration

| Parameter     | Value           |
| ------------- | --------------- |
| Batch Size    | 16              |
| Epochs        | 2               |
| Learning Rate | 1e-4            |
| Evaluation    | Every 500 steps |
| Metric        | Accuracy        |

---

## 📊 Evaluation Metric

The model is evaluated using **classification accuracy**:

```python
accuracy = correct_predictions / total_predictions
```

---

## 🏋️ Training

To start fine-tuning the model:

```bash
python train.py
```

The best-performing model checkpoint is automatically saved.

---

## 📌 Key Features

✅ Vision Transformer–based classification
✅ Transfer learning with frozen backbone
✅ Clean custom PyTorch Dataset
✅ Hugging Face Trainer integration
✅ Modular and extensible design

---

## 🧠 Possible Improvements

* Gradual layer unfreezing
* Class imbalance handling
* Precision / Recall / F1-score metrics
* Integration with Streamlit or FastAPI
* RAG-based explainability layer

---

## 👨‍💻 Author

**Abin**
Full-Stack AI Engineer | ML | DL | NLP | Vision

---

## 📜 License

This project is intended for **educational and research purposes**.

---

⭐ If you like this project, give it a star and feel free to contribute!
