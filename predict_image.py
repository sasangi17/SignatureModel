
from pathlib import Path
from PIL import Image
import tensorflow as tf
import numpy as np

MODEL_PATH = Path("outputs/best_signature_model.keras")
IMAGE_FOLDER = Path("real_world_samples")

IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5

print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_paths = [
    path for path in IMAGE_FOLDER.iterdir()
    if path.is_file() and path.suffix.lower() in extensions
]

if not image_paths:
    print("No images found.")
    raise SystemExit

print("\nTesting images with white background handling...\n")

for image_path in image_paths:
    try:
        with Image.open(image_path) as original:
            image = original.convert("RGBA")

            # Place transparent image areas on a white background
            white_background = Image.new(
                "RGBA",
                image.size,
                (255, 255, 255, 255)
            )

            image = Image.alpha_composite(
                white_background,
                image
            ).convert("RGB")

            image = image.resize(IMAGE_SIZE)

            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)
        probability = float(prediction[0][0])

        result = (
            "SIGNATURE"
            if probability >= THRESHOLD
            else "NOT A SIGNATURE"
        )

        print(f"Image: {image_path.name}")
        print(f"Prediction: {result}")
        print(f"Signature probability: {probability:.2%}")
        print("-" * 45)

    except Exception as error:
        print(f"Error with {image_path.name}: {error}")
