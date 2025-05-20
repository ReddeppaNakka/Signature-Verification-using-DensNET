# 🧠 Signature Verification using DenseNet169

This project aims to verify the authenticity of handwritten signatures using a deep learning model based on **DenseNet169**. The model classifies signature images as either **real** or **forged**, helping automate the signature verification process in domains like banking, legal documentation, and identity verification.

---

## 📁 Dataset Structure

Organize your dataset as follows:


- `real/`: Genuine signatures
- `forged/`: Imitated/fake signatures

---

## 🔧 Features

- Uses pre-trained **DenseNet169** for robust feature extraction
- Custom classification head for binary classification
- Preprocessing with Keras `ImageDataGenerator`
- Model training, validation, and testing with visualization
- Predicts class label and confidence score for a given test image

---

## 🛠️ How It Works

### 1. Preprocessing
Images are resized to 224x224 and preprocessed using DenseNet-specific preprocessing.

### 2. Model Architecture
- **Base**: DenseNet169 (without top)
- **Added Layers**:
  - `GlobalAveragePooling2D`
  - `Dense(1024, relu)`
  - `Dense(2, softmax)`

### 3. Training
- Optimizer: Adam (learning rate = 0.001)
- Loss: Categorical Crossentropy
- Metrics: Accuracy

### 4. Testing
Loads a sample image and outputs:
- Predicted class (`real` or `forged`)
- Confidence (%)

---

## 🖼️ Example Output

![Prediction Example](https://via.placeholder.com/500x300.png?text=Predicted:+Real,+Accuracy:+98.45%)

---

## 📈 Future Enhancements

- Train for more epochs for better generalization
- Apply data augmentation (rotation, zoom, etc.)
- Save and load models with `model.save()` and `load_model()`
- Add performance metrics like F1-score, ROC-AUC
- Compare with other architectures (e.g., ResNet, EfficientNet)

---

## 📦 Dependencies

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Keras

Install using:
```bash
pip install tensorflow numpy matplotlib
# Clone this repo
git clone https://github.com/Reddeppanakka/signature-verification-densenet.git
cd signature-verification-densenet

# Place your dataset inside the 'data/' directory

# Run the main script
python signature_verification.py
