from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).parent

MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "signature_classifier.keras"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "tensorflowjs"
)

if not MODEL_PATH.exists():

    print("Trained model not found.")

    print("Train the model first.")

    sys.exit(1)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

command = [

    "tensorflowjs_converter",

    "--input_format=keras",

    str(MODEL_PATH),

    str(OUTPUT_DIR)
]

print("Converting Keras model...")

result = subprocess.run(
    command,
    check=False
)

if result.returncode == 0:

    print("\nConversion successful!")

    print("Output:", OUTPUT_DIR)

else:

    print("\nConversion failed.")

    print(
        "We will resolve TensorFlow.js "
        "compatibility before continuing."
    )
