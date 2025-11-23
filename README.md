Breast Cancer Prediction using CNN

A Convolutional Neural Network (CNN)–based project for classifying benign vs malignant breast cancer histopathology images.

This repository includes:

Scripts for dataset preparation

CNN model training (PyTorch/Keras — based on your latest code it's PyTorch)

Prediction script

Clean project structure

Dataset workflow

GPU/CPU instructions

🚀 Clone the Repository
git clone https://github.com/AbrahamNevin/Breast-Cancer-Prediction-using-CNN-
cd Breast-Cancer-Prediction-using-CNN-

📥 Download the Dataset

This project uses the Breast Histopathology Images dataset from Kaggle:

🔗 Dataset Link:
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

After downloading, extract it and place it inside:

data/raw/


The final folder MUST look like:

data/raw/IDC_regular_ps50_idx5/


⚠️ This folder contains 277,524 images inside class folders 0 and 1.
It is very large — so it is intentionally ignored by GitHub (.gitignore).

📁 Project Structure
Breast-Cancer-Prediction-using-CNN-
│
├── data/
│   ├── raw/
│   │   └── IDC_regular_ps50_idx5/
│   │       ├── 0/   (benign images)
│   │       └── 1/   (malignant images)
│   │
│   ├── processed/
│       ├── train/
│       │   ├── 0/
│       │   └── 1/
│       ├── val/
│       │   ├── 0/
│       │   └── 1/
│       └── test/
│           ├── 0/
│           └── 1/
│
├── models/
│   └── best_model.pth           # Saved model
│
├── scripts/
│   ├── split_dataset.py         # Splits raw images into train/val/test
│   ├── train_model.py           # Train CNN model
│   ├── predict.py               # Predict on a single image
│   └── utils.py
│
├── notebooks/                   # (optional Jupyter work)
│
├── requirements.txt
└── README.md

🔧 Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


If PyTorch is missing:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

🛠️ Step 1 — Split the Dataset

Run after placing Kaggle dataset in data/raw/:

python scripts/split_dataset.py


This will create train, val, and test sets:

data/processed/train/
data/processed/val/
data/processed/test/

🧠 Step 2 — Train the CNN
python scripts/train_model.py


The trained model will be saved inside:

models/best_model.pth

🔍 Step 3 — Make Predictions
python scripts/predict.py --image path/to/image.png

📌 Notes

GPU is optional — the project works on CPU.

Do NOT upload dataset to GitHub (too large).

.gitignore excludes dataset folders automatically.

❤️ Credits

Dataset by:
Paul Timothy Mooney
Kaggle — Breast Histopathology Images

Project by:
Nevin Abraham
