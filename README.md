# Breast Cancer Prediction using CNN

A Convolutional Neural Network (CNN)–based project for classifying benign vs malignant breast cancer histopathology images.

This repository includes:

- Scripts for dataset preparation  
- CNN model training (PyTorch)  
- Prediction script  
- Clean project structure  
- Dataset workflow  
- GPU/CPU instructions  

---

## 🚀 Clone the Repository

```bash
git clone https://github.com/AbrahamNevin/Breast-Cancer-Prediction-using-CNN-
cd Breast-Cancer-Prediction-using-CNN-
```

---

## 🔥 Download the Dataset

This project uses the Breast Histopathology Images dataset from Kaggle:

**Dataset Link:**  
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

After downloading, extract it and place it exactly inside:

```
data/raw/
```

The final folder MUST look like:

```
data/raw/IDC_regular_ps50_idx5/
```

⚠️ This folder contains **277,524 images** inside class folders `0` and `1`.  
It is very large — therefore it is intentionally ignored by Git (`.gitignore`).

---

## 📁 Project Structure

```
Breast-Cancer-Prediction-using-CNN-
│── data/
│   ├── raw/
│   │   └── IDC_regular_ps50_idx5/
│   │       ├── 0/   (benign images)
│   │       └── 1/   (malignant images)
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│
│── models/
│   └── best_model.pth
│
│── scripts/
│   ├── split_dataset.py
│   └── train_model.py
│
│── predict.py
│── utils.py
│── notebooks/
│── requirements.txt
│── README.md
```

---

## 🧪 Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python scripts/train_model.py
```

---

## 📊 Predict on a Single Image

```bash
python predict.py --image path/to/image.png
```

---

## 🎯 Notes

- The dataset must be placed exactly inside `data/raw/`.  
- The `scripts/split_dataset.py` file automatically creates train/val/test folders.  
- `best_model.pth` will be generated inside `models/` after training.

---
