
import random
import shutil
from pathlib import Path

import tensorflow as tf
from PIL import Image

# Project folders
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_dataset"
DATASET_DIR = BASE_DIR / "dataset_ready"

# Reproducibility
SEED = 42
random.seed(SEED)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Collect genuine and forged signature images
signature_files = []

for image_path in RAW_DIR.iterdir():
    if not image_path.is_file():
        continue

    if image_path.name.startswith(("original_", "forgeries_")):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                signature_files.append(image_path)
            except Exception:
                print(f"Skipping invalid image: {image_path.name}")

print("Signature images found:", len(signature_files))

if not signature_files:
    raise RuntimeError("No signature images found in raw_dataset.")

# Download CIFAR-10 for preliminary negative examples
print("Loading CIFAR-10 dataset...")

(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

negative_images = list(x_train[:1800]) + list(x_test[:200])
random.shuffle(negative_images)

# Split images into train, validation and test
random.shuffle(signature_files)

def split_items(items):
    total = len(items)
    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    return (
        items[:train_end],
        items[train_end:validation_end],
        items[validation_end:]
    )

signature_splits = split_items(signature_files)
negative_splits = split_items(negative_images)



for split in ("train", "validation", "test"):
    for category in ("signature", "not_signature"):
        (DATASET_DIR / split / category).mkdir(
            parents=True,
            exist_ok=True
        )

def save_signature_images(files, split):
    destination = DATASET_DIR / split / "signature"

    for index, source in enumerate(files):
        try:
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.save(destination / f"signature_{index:05d}.jpg",
                           quality=95)
        except Exception as error:
            print(f"Could not process {source.name}: {error}")

def save_negative_images(images, split):
    destination = DATASET_DIR / split / "not_signature"

    for index, image_array in enumerate(images):
        image = Image.fromarray(image_array)
        image.save(destination / f"negative_{index:05d}.jpg",
                   quality=95)

# Save all splits
for split, signatures, negatives in zip(
    ("train", "validation", "test"),
    signature_splits,
    negative_splits
):
    save_signature_images(signatures, split)
    save_negative_images(negatives, split)

    print(f"\n{split.upper()}")
    print("Signatures:", len(signatures))
    print("Not signatures:", len(negatives))

print("\nDataset preparation completed.")
print("Dataset location:", DATASET_DIR)
