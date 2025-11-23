import tensorflow as tf
from utils import load_image

MODEL_PATH = "models/best_model.h5"

def predict(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    img = load_image(image_path)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        print(f"{image_path} → MALIGNANT ({pred:.2f})")
    else:
        print(f"{image_path} → BENIGN ({pred:.2f})")

if __name__ == "__main__":
    # Example
    path = input("Enter image path: ")
    predict(path)
