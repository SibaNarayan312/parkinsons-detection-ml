import os
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

image_dir = "images"
metadata_path = "spiral.xlsx"

if not os.path.exists(image_dir):
    raise FileNotFoundError("Image folder not found! Please place PNG images inside images/")

if not os.path.exists(metadata_path):
    raise FileNotFoundError("spiral.xlsx not found!")

metadata = pd.read_excel(r"spiral.xlsx")
metadata = metadata[["Sample", "Status"]].dropna()

features = []
labels = []

print("Starting image feature extraction...")
for _, row in metadata.iterrows():
    file_name = str(row["Sample"]).strip()
    if not file_name:
        print("Skipping row with missing Sample name.")
        continue

    img_path = os.path.join(image_dir, file_name + ".png")
    if not os.path.exists(img_path):
        print(f"Image not found for sample: {file_name}")
        continue

    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        edges = cv2.Canny(img, 50, 150)
        feature_vector = edges.flatten()
        features.append(feature_vector)
        labels.append(int(row["Status"]))
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

if not features:
    raise ValueError("No image features extracted. Ensure filenames in spiral.xlsx match .png files in images/")

print("Feature extraction complete.")

X = np.array(features)
Y = np.array(labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)

print("Training model...")
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, Y_train)

train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))

print(f"Training Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/spiral_model.sav", "wb"))
pickle.dump(scaler, open("models/spiral_scaler.sav", "wb"))

print("Spiral model and scaler saved successfully in 'models/' folder.")

example_image_path = "images/V02HE02.png"

if os.path.exists(example_image_path):
    img = cv2.imread(example_image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    edges = cv2.Canny(img, 50, 150)
    feature_vector = edges.flatten().reshape(1, -1)
    input_scaled = scaler.transform(feature_vector)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        print(f"Prediction: Parkinson’s detected with {probability*100:.2f}% confidence.")
    else:
        print(f"Prediction: Healthy spiral detected with {(1 - probability)*100:.2f}% confidence.")
else:
    print(f"\nExample image file not found at: {example_image_path}")
    print("Skipping example prediction.")