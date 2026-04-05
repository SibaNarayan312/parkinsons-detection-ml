# 🧠 Parkinson’s Disease Detection using Machine Learning

This project focuses on detecting Parkinson’s Disease using multiple data modalities including **audio signals, spiral drawings, video gait analysis, and tabular biomedical data**.

---

## 🚀 Overview

Parkinson’s Disease is a neurodegenerative disorder that affects movement, speech, and motor control. Early detection can help in better management and treatment.

This project uses Machine Learning models to analyze:

- 🎤 Voice recordings (Audio)
- ✍️ Spiral drawings (Image)
- 🚶 Walking patterns (Video)
- 📊 Biomedical measurements (Tabular data)

---

## 🧩 Modules in the Project

### 1️⃣ Audio Analysis (`audio.py`)
- Extracts **MFCC (Mel Frequency Cepstral Coefficients)** features
- Uses SVM classifier for prediction
- Detects voice irregularities caused by Parkinson’s

---

### 2️⃣ Spiral Image Analysis (`spiral.py`)
- Uses **edge detection (Canny)**
- Converts images into feature vectors
- Identifies tremor patterns in drawings

---

### 3️⃣ Video Gait Analysis (`video.py`)
- Uses **MediaPipe Pose estimation**
- Extracts body movement keypoints
- Detects abnormalities in walking patterns

---

### 4️⃣ Tabular Data Model (`linear.py`)
- Uses biomedical voice measurements
- Applies **Linear SVM**
- Predicts based on structured data

---

## ⚙️ Technologies Used

- Python 🐍
- NumPy, Pandas
- Scikit-learn
- OpenCV
- Librosa (Audio processing)
- MediaPipe (Pose detection)

---

## 🧪 Machine Learning Model

- Algorithm: **Support Vector Machine (SVM)**
- Kernel: RBF (for complex data patterns)
- Data split: 80% training, 20% testing
- Feature scaling: StandardScaler

---

## 📊 Workflow

1. Data Collection
2. Feature Extraction (Audio/Image/Video)
3. Data Preprocessing
4. Model Training (SVM)
5. Evaluation (Accuracy)
6. Prediction
