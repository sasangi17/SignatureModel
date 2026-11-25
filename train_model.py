import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_ready"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

EPOCHS = 15

SEED = 42

CLASS_NAMES = [
    "not_signature",
    "signature"
]

# CPU settings
tf.keras.utils.set_random_seed(SEED)

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading training dataset...")

train_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

print("Loading validation dataset...")

validation_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "validation",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# ==========================================
# PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

validation_ds = validation_ds.prefetch(
    AUTOTUNE
)

# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = keras.Sequential([

    layers.RandomRotation(
        0.03
    ),

    layers.RandomZoom(
        0.08
    ),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )

], name="data_augmentation")

# ==========================================
# PRETRAINED MODEL
# ==========================================

print("Loading MobileNetV2...")

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ==========================================
# BUILD MODEL
# ==========================================

inputs = keras.Input(
    shape=(224, 224, 3),
    name="image"
)

x = data_augmentation(inputs)

x = keras.applications.mobilenet_v2.preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="signature_probability"
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs
)

# ==========================================
# COMPILE
# ==========================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        )
    ]
)

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

callbacks = [

    keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=4,

        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(

        filepath=str(
            OUTPUT_DIR / "best_signature_model.keras"
        ),

        monitor="val_loss",

        save_best_only=True,

        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=0.00001,

        verbose=1
    )
]

# ==========================================
# TRAIN
# ==========================================

print("\nStarting training...")

history = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    callbacks=callbacks
)

# ==========================================
# SAVE FINAL MODEL
# ==========================================

final_model_path = (
    OUTPUT_DIR / "signature_classifier.keras"
)

model.save(final_model_path)

print(
    "\nModel saved:",
    final_model_path
)

# ==========================================
# SAVE HISTORY
# ==========================================

history_path = (
    OUTPUT_DIR / "training_history.json"
)

history_data = {
    key: [
        float(value)
        for value in values
    ]

    for key, values in history.history.items()
}

with open(
    history_path,
    "w"
) as file:

    json.dump(
        history_data,
        file,
        indent=4
    )

print(
    "Training history saved:",
    history_path
)

print("\nTRAINING COMPLETED")
