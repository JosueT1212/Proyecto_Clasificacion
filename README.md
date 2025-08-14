# Data Analysis for Medical Applications

## 📄 Project Overview
This project implements a **machine learning–based activity recognition system** designed for medical and fitness applications.  
Using **accelerometer signals** from a smartphone, the system classifies physical activities such as:
- Squats
- Push-ups
- Stair climbing / descending
- Crunches
- Jumping jacks

The model was trained and evaluated using **Scikit-learn** and **NumPy**, applying **feature engineering**, **model comparison**, **hyperparameter tuning**, and **real-time classification**.

---

## 🔍 Data & Feature Extraction
The accelerometer data is collected in **x**, **y**, and **z** axes using **Phyphox**.  
From these signals, the following **time-domain features** are extracted:

- **Mean**
- **Standard Deviation**
- **Kurtosis**
- **Skewness**
- **Median Absolute Deviation (MAD)**
- **Root Mean Square (RMS)**
- **Zero Crossing Rate (ZCR)**
- **Slope Sign Changes (SSC)**
- **Waveform Length (WL)**
- **Signal Energy**

These features serve as inputs for the classification algorithms.

---

## 🤖 Model Evaluation
We compared multiple classification algorithms:

| Model                              | Accuracy |
|------------------------------------|----------|
| Linear Discriminant Analysis (LDA) | **94%**  |
| Gaussian Naive Bayes               | **93%**  |
| Linear SVM                         | 89%      |
| K-Nearest Neighbors (KNN)          | 88%      |
| Gradient Boosting                  | 89%      |
| Multi-layer Perceptron (MLP)       | 88%      |
| Ridge Classifier                    | 84%      |
| Nearest Centroid                   | 73%      |
| SVM (RBF kernel)                   | 64%      |
| Stochastic Gradient Descent        | 52%      |

---

## ⚙️ Optimization
We optimized the **Linear SVM** and **GaussianNB** models through:
- **Hyperparameter tuning**:
  - SVM: Optimal `C ≈ 0.4`
  - GaussianNB: Optimal `var_smoothing ≈ 1e-9`
- **Feature selection**: `SelectKBest` reduced dimensionality to **19–21 optimal features**
- **Nested cross-validation** for robust performance estimation

**Optimized results:**
- Linear SVM → **91% accuracy**
- GaussianNB → **93% accuracy**

---

## 🚀 Real-Time Application
The **GaussianNB (optimized)** model was deployed for real-time activity classification.  

**Pipeline:**
1. Receive accelerometer signals in real-time.
2. Apply feature extraction.
3. Transform data using the selected feature set.
4. Predict activity using the trained model.

**Performance:**
- ~93% accuracy in cross-validation
- ~90% correct predictions during live testing
- Some misclassifications occur when activity intensity is low

---

## 🛠 Technologies
- Python 3
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Phyphox (data acquisition)

---

## 📦 Installation
```bash
git clone https://github.com/yourusername/medical-activity-recognition.git
cd medical-activity-recognition
pip install -r requirements.txt
