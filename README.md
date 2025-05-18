# Chest-X-ray-knowledge-distillation-or-fine-tuning-
This project compares three deep learning models—custom CNN, Fine tuned ResNet-50, and a teacher-student distillation model—for classifying pneumonia from chest X-ray images. It reveals performance trade-offs in accuracy, recall, and efficiency to guide model selection in medical AI.
# 🩺 Pneumonia Detection from Chest X-Rays using Deep Learning

This project explores and compares three deep learning approaches—**a custom CNN**, **ResNet-50 via transfer learning**, and **a teacher-student model using knowledge distillation**—for binary classification of chest X-ray images as **Pneumonia** or **Normal**.

---
- **Source**: [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-x-ray-pneumonia)
- **Size**: 5,187 labeled X-ray images (pediatric patients)
- **Classes**: Pneumonia vs Normal
- **Preprocessing**:
  - Converted to grayscale
  - Resized to 224×224 pixels
  - Normalized pixel values to [0, 1]
  - Data augmentation: flipping, rotating, zooming

---

## 🧠 Models Overview

### 1. Base CNN (Custom)
- 11 convolutional layers 5 fully connected, max-pooling, batch norm, dropout, and a fully connected output
- **Test Accuracy**: 91.19%
- **AUC-ROC**: 0.9073
- Performs best in **overall accuracy** and **balanced performance**

### 2. Teacher-Student Model (Knowledge Distillation)
- A lightweight teacher model with 3 CNN, 3 dense layers
- A smaller student model also with 2CNN, 3 dense layers
- Despite its simplicity, the student model **outperformed the fine-tuned ResNet-50** in recall and AUC
- **Student Accuracy**: 81.09%  
- **Recall**: 92.56%  
- **AUC-ROC**: 0.9036  
- ⚠️ High recall is crucial in medical applications to reduce false negatives

#### 📐 Distillation Loss Function

To train the student model, I used a **simple yet powerful** combination of:

- **Binary Cross-Entropy Loss**:
  \[
  L_{\text{BCE}} = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
  \]

- **Kullback-Leibler Divergence** between the soft predictions of teacher and student:
  \[
  L_{\text{KL}} = \sum_i P(i) \log \left(\frac{P(i)}{Q(i)}\right)
  \]

This approach was selected because it balances **interpretability**, **ease of implementation**, and **low resource demand**, making it suitable for real-world deployment in healthcare.

###  3. Transfer Learning (ResNet-50)
- Fine-tuned ImageNet-pretrained ResNet-50
- **Test Accuracy**: 87.5%  
- **Precision & F1-score**: Highest among all
- Great at reducing **false positives**

---

## 📊 Model Comparison Summary

| Model             | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Notable Strength       |
|------------------|----------|-----------|--------|----------|---------|------------------------|
| Base CNN         | 91.19%   | 0.88      | 0.89   | 0.88     | 0.91    | Best overall accuracy  |
| Student Model    | 81.09%   | 0.79      | 0.93   | 0.85     | 0.90    | Best recall (low FN)   |
| ResNet-50        | 87.50%   | 0.90      | 0.90   | 0.90     | 0.87    | Best precision & F1    |

---

##  Tools & Libraries

- Python
- PyTorch / TensorFlow (depending on implementation)
- OpenCV / PIL for image processing
- Matplotlib & Seaborn for visualization

---
