# ✍️ Handwritten Signature Classification Model

A deep learning-based image classification system developed using **TensorFlow and MobileNetV2** to identify whether an uploaded image contains a handwritten signature or is not a signature.

The model uses transfer learning with a pretrained MobileNetV2 architecture and is designed to support signature image classification in document-processing workflows.

## 🚀 Project Overview

This project focuses on automatically classifying images into two categories:

- **Signature** – Images containing handwritten signatures.
- **Not Signature** – Images that do not contain handwritten signatures.

The system uses a Convolutional Neural Network (CNN) with MobileNetV2 transfer learning to extract image features and perform binary classification.

## ✨ Key Features

- Handwritten signature image classification.
- Pretrained MobileNetV2 architecture.
- Transfer learning using ImageNet weights.
- Image preprocessing and dataset preparation.
- Data augmentation to improve model generalization.
- Training with early stopping and learning rate reduction.
- Model evaluation using classification metrics.
- Individual image prediction.
- TensorFlow.js model conversion for potential frontend integration.

## 🧠 Model Architecture

**MobileNetV2 – Transfer Learning**

The model uses MobileNetV2 as its feature extraction backbone.

Architecture:

1. Input Layer – 224 × 224 RGB image.
2. Data Augmentation – Random rotation, zoom, and translation.
3. Image Preprocessing – MobileNetV2 preprocessing.
4. MobileNetV2 – Pretrained ImageNet feature extraction.
5. Global Average Pooling.
6. Dropout – 0.30.
7. Dense Output Layer – Sigmoid activation.

**Classification Output:**
- `0` – Not Signature
- `1` – Signature

## 🛠️ Technologies Used

- Python
- TensorFlow 2.16.1
- Keras
- MobileNetV2
- NumPy
- Pillow
- Scikit-learn
- TensorFlow.js (for model conversion)

## 📂 Project Structure

```text
SignatureModel/
│
├── outputs/
│   ├── best_signature_model.keras
│   ├── signature_classifier.keras
│   ├── training_history.json
│   └── evaluation_results.txt
│
├── prepare_dataset.py
├── train_model.py
├── evaluate_model.py
├── predict_image.py
├── convert_model.py
├── download_dataset.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** Dataset folders and virtual environments are excluded from Git.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sasangi17/SignatureModel.git

cd SignatureModel
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

The model uses two image categories:

- Signature images collected from handwritten signature samples.
- Not-signature images obtained from the CIFAR-10 dataset as negative examples.

Place the signature images inside the `raw_dataset/` directory.

Supported image formats:

- JPG
- JPEG
- PNG
- BMP
- WEBP

Run the dataset preparation script:

```bash
python prepare_dataset.py
```

The script automatically:

1. Identifies valid signature images.
2. Loads negative image samples from CIFAR-10.
3. Splits the dataset into training, validation, and testing sets.
4. Organizes images into the appropriate class folders.

Dataset split:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

## 🏋️ Model Training

Run the following command to train the model:

```bash
python train_model.py
```

Training configuration:

| Parameter | Value |
|---|---|
| Architecture | MobileNetV2 |
| Input Image Size | 224 × 224 |
| Batch Size | 16 |
| Epochs | 15 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Crossentropy |
| Dropout | 0.30 |
| Random Seed | 42 |

The training process saves:

- `best_signature_model.keras` – Best model based on validation loss.
- `signature_classifier.keras` – Final trained model.
- `training_history.json` – Training and validation metrics.

## 📈 Model Evaluation

Evaluate the trained model using the test dataset:

```bash
python evaluate_model.py
```

The evaluation process generates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Evaluation Results

The currently recorded evaluation results are:

| Metric | Result |
|---|---:|
| Accuracy | 100% |
| Precision (Not Signature) | 100% |
| Recall (Not Signature) | 100% |
| F1-score (Not Signature) | 100% |
| Precision (Signature) | 100% |
| Recall (Signature) | 100% |
| F1-score (Signature) | 100% |

**Test Dataset:**
- Not Signature: 300 images
- Signature: 396 images
- Total: 696 images

These results represent the current test dataset and should not be interpreted as guaranteed real-world accuracy.

## 🔍 Image Prediction

To classify individual signature images, create a folder named `real_world_samples/` and place the images inside it.

Run:

```bash
python predict_image.py
```

The prediction script:

1. Loads the trained model.
2. Reads images from the sample folder.
3. Handles transparent image backgrounds.
4. Resizes images to 224 × 224.
5. Predicts the classification.
6. Displays the signature probability.

Example output:

```text
Image: sample_signature.jpg

Prediction: SIGNATURE

Signature probability: 98.50%
```

*Example output is illustrative.*

## 🌐 TensorFlow.js Conversion

The project includes a script to convert the trained Keras model into TensorFlow.js format for potential integration with web applications.

Run:

```bash
pip install tensorflowjs
```

Then execute:

```bash
python convert_model.py
```

The converted model will be saved inside:

```text
outputs/tensorflowjs/
```

## 🔮 Future Improvements

- Improve classification performance using more diverse real-world images.
- Expand the negative image dataset.
- Test performance on scanned documents and different signature backgrounds.
- Integrate the trained model into web applications.
- Develop a frontend signature classification interface.

## ⚠️ Important Note

This model identifies whether an image contains a handwritten signature. It does **not** verify whether a signature is genuine or forged, identify the signer, or establish the authenticity of a document.

A separate signature verification model would be required for those tasks.

## 👩‍💻 Author

**Sasangi Samadara**

- GitHub: [@sasangi17](https://github.com/sasangi17)

---

*Developed as a deep learning project using Python, TensorFlow, and MobileNetV2.*
