import os
import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ----------- PATHS (FIXED) -----------
video_dir = "Videos"
metadata_path = "PDFEinfo_cleaned_wide.csv"

if not os.path.exists(video_dir):
    raise FileNotFoundError("❌ Video folder not found!")

if not os.path.exists(metadata_path):
    raise FileNotFoundError("❌ Metadata file not found!")

# ----------- LOAD METADATA -----------
metadata = pd.read_csv(metadata_path)
metadata = metadata[["Sample", "Status"]].dropna()


print(f"Loaded metadata with {len(metadata)} entries.")
print(f"Unique Status values: {metadata['Status'].unique()}")

# ----------- MEDIAPIPE SETUP -----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

features = []
labels = []

print("\nStarting video feature extraction...\n")

# ----------- PROCESS VIDEOS -----------
for _, row in metadata.iterrows():
    sample = str(row["Sample"]).strip()
    file_path = os.path.join(video_dir, sample + ".mp4")

    print(f"Checking: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ Missing file: {file_path}")
        continue

    cap = cv2.VideoCapture(file_path)
    frame_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            coords = []
            for lm in results.pose_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            frame_features.append(coords)

    cap.release()

    if frame_features:
        avg_features = np.mean(frame_features, axis=0)
        features.append(avg_features)
        labels.append(int(row["Status"]))
        print(f"✅ Processed: {sample}")
    else:
        print(f"⚠️ No landmarks detected in: {sample}")

# ----------- CHECK FEATURES -----------
if not features:
    raise ValueError("❌ No features extracted. Check video files!")

print("\nFeature extraction complete.")

# ----------- ML PART -----------
X = np.array(features)
Y = np.array(labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=2
)

print("\nTraining model...")
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, Y_train)

train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))

print(f"\nTraining Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")
p
# ----------- SAVE MODEL -----------
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/video_model.sav", "wb"))
pickle.dump(scaler, open("models/video_scaler.sav", "wb"))

print("\n✅ Model and scaler saved in 'models/' folder.")

# ----------- TEST ON ONE VIDEO -----------
test_video = os.path.join(video_dir, metadata.iloc[0]["Sample"] + ".mp4")

if os.path.exists(test_video):
    print("\n--- Running Example Prediction ---")
    cap = cv2.VideoCapture(test_video)
    frame_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            coords = []
            for lm in results.pose_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            frame_features.append(coords)

    cap.release()

    if frame_features:
        avg_features = np.mean(frame_features, axis=0).reshape(1, -1)
        input_scaled = scaler.transform(avg_features)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction[0] == 1:
            print(f"Prediction: Parkinson’s detected ({probability*100:.2f}%)")
        else:
            print(f"Prediction: Healthy ({(1-probability)*100:.2f}%)")
else:
    print("⚠️ Test video not found.")