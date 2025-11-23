📌 Breast Cancer Prediction using CNN

A Convolutional Neural Network (CNN) project for classifying benign vs malignant breast cancer histopathology images.

This repository includes:

🗂 Scripts for dataset preparation

🧠 CNN model training (PyTorch)

🔍 Prediction script

🧼 Clean project structure

📁 Dataset workflow

⚙️ GPU/CPU instructions

📥 Clone the Repository
git clone https://github.com/AbrahamNevin/Breast-Cancer-Prediction-using-CNN-
cd Breast-Cancer-Prediction-using-CNN-

🔥 Download the Dataset

This project uses the Breast Histopathology Images dataset from Kaggle.

🔗 Dataset Link:
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

After downloading, extract the .zip file and place the folder inside:

data/raw/

📁 Required Dataset Structure

After extraction, your folder must look like this:

data/
└── raw/
    └── IDC_regular_ps50_idx5/
        ├── 0/        (benign images)
        └── 1/        (malignant images)


⚠️ This folder contains 277,524 images, so it is intentionally ignored by GitHub (.gitignore).

📂 Project Structure
Breast-Cancer-Prediction-using-CNN-
│
├── data/
│   ├── raw/
│   │   └── IDC_regular_ps50_idx5/
│   ├── processed/
│   │   ├── train/
│   │   │   ├── 0/
│   │   │   └── 1/
│   │   ├── val/
│   │   │   ├── 0/
│   │   │   └── 1/
│   │   └── test/
│   │       ├── 0/
│   │       └── 1/
│
├── models/
│   └── best_model.pth           # Saved PyTorch model
│
├── scripts/
│   ├── split_dataset.py         # Splits raw images into train/val/test
│   ├── train_model.py           # Train CNN model
│   └── predict.py               # Predict on a single image
│
├── utils.py
├── requirements.txt
└── README.md

⚙️ Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

🏋️ Train the Model

Make sure the dataset is placed correctly (as shown above), then run:

python scripts/split_dataset.py
python scripts/train_model.py

🔍 Run Prediction on a Single Image
python scripts/predict.py --image path/to/image.png

📌 Notes

You can train on CPU, but training is much faster on GPU (CUDA required).

Dataset is very large — ensure you have enough storage.

Processed images (train/val/test) will be created automatically.
