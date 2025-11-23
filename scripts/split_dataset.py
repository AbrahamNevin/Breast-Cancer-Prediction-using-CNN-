import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw/IDC_regular_ps50_idx5"
OUTPUT_DIR = "data/processed"

# Create output directories
train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "val")
test_dir = os.path.join(OUTPUT_DIR, "test")

for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

# Collect all images recursively
all_images = []
for root, dirs, files in os.walk(RAW_DIR):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            all_images.append(os.path.join(root, file))

print("Total images found:", len(all_images))

if len(all_images) == 0:
    raise ValueError("No images found! Check dataset path or folder structure.")

# Split into train 70%, temp 30%
train_files, temp_files = train_test_split(all_images, test_size=0.30, random_state=42)

# Split temp into val 15% and test 15%
val_files, test_files = train_test_split(temp_files, test_size=0.50, random_state=42)

def copy_files(file_list, dest_dir):
    for f in file_list:
        # Maintain class folder (0 or 1)
        class_label = os.path.basename(os.path.dirname(f))
        target_folder = os.path.join(dest_dir, class_label)
        os.makedirs(target_folder, exist_ok=True)

        shutil.copy2(f, target_folder)

print("Copying training images...")
copy_files(train_files, train_dir)

print("Copying validation images...")
copy_files(val_files, val_dir)

print("Copying test images...")
copy_files(test_files, test_dir)

print("Dataset split complete!")
print(f"Train: {len(train_files)} images")
print(f"Val:   {len(val_files)} images")
print(f"Test:  {len(test_files)} images")
