# Breast Cancer Classification using CNN

A deep learning project that classifies breast ultrasound images into Benign and Malignant categories using Convolutional Neural Networks (CNNs) built with TensorFlow/Keras.

📦 Dataset
Source

The dataset is from Kaggle:
👉[Breast Ultrasound Images Dataset on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset?utm_source=chatgpt.com)


The original dataset contained three classes:

benign

malignant

normal

For this project, only benign and malignant images were used.

🧹 Dataset Preprocessing

A preprocessing script (pre_processing.ipynb) was used to:

Remove the normal class

Split the remaining data into train (70%), validation (15%), and test (15%)

Organize images into a new folder structure:

breast_cancer_dataset/
│
├── train/
│   ├── benign/
│   └── malignant/
│
├── val/
│   ├── benign/
│   └── malignant/
│
└── test/
    ├── benign/
    └── malignant/

📜 Preprocessing Script (Summary)
# Key functionality:
- Reads images from "breast_ultrasound_images"
- Shuffles and splits into train/val/test
- Copies to "breast_cancer_dataset" with organized folders
- Prints "✅ Dataset reorganized successfully!"

Models Used

This project experiments with three different Convolutional Neural Network (CNN) architectures for breast cancer ultrasound image classification (Benign vs Malignant).

🩺 Model 1 – Baseline CNN

A simple 3-block CNN model without regularization, used as the baseline.

Architecture:

3 Convolution + ReLU + MaxPooling blocks

Dropout after the last convolutional block (0.3)

Dense layer (128 units) + Dropout (0.5)

Output layer: Sigmoid activation

Compile settings:

optimizer = Adam(learning_rate=learning_rate)
loss = 'binary_crossentropy'
metrics = ['accuracy']

🧩 Model 2 – Regularized CNN

An improved version of Model 1 with L2 regularization and higher dropout to reduce overfitting.

Enhancements:

L2 regularization (0.001) applied to Conv and Dense layers

Increased dropout rates (0.4, 0.5)

Same convolutional structure as Model 1

Purpose:
To test whether explicit regularization helps improve generalization on unseen data.

⚙️ Model 1_1 – Hybrid (Dropout + Light Regularization)

A tuned version combining the strengths of Model 1 and Model 2 with balanced dropout and smaller L2 regularization.

Features:

Added dropout (0.2–0.3) after each convolutional block

Light L2 regularization (0.0005) in the dense layer

Lower learning rate (0.0001) for stable convergence

Goal:
To balance underfitting and overfitting for better test accuracy stability.

### 📊 Comparison Summary

| Model     | Regularization | Dropout | Learning Rate     | Test Accuracy |
|------------|----------------|----------|-------------------|----------------|
| Model 1   | ❌ No            | 0.3–0.5  | Default (`0.001`) | ~0.82 |
| Model 2   | ✅ L2(0.001)     | 0.4–0.5  | Default (`0.001`) | ~0.77 |
| Model 1_1 | ✅ L2(0.0005)    | 0.2–0.5  | `0.0001`          | ~0.75 |

🧩 Technologies Used

Python 🐍

TensorFlow / Keras

NumPy

Matplotlib

scikit-learn

## 📊 Results

### 🖼️ Visualization
- Displayed **original vs. resized ultrasound images** to verify preprocessing quality.  
- Confirmed that resized images preserved key tumor features.

### 📈 Model Performance
- **Training & Validation Curves:**  
  Plotted accuracy and loss trends to monitor convergence and overfitting.  
- **Test Set Evaluation:**  
  Reported model accuracy, demonstrating strong generalization on unseen data.
  
