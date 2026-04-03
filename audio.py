import os
import librosa
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

audio_dir = r"C:\Users\HP\OneDrive\Desktop\New folder\HC_AH"
metadata_path = r"C:\Users\HP\OneDrive\Desktop\New folder\Demographics_age_sex.xlsx"

if not os.path.exists(audio_dir):
    raise FileNotFoundError("Audio folder not found! Please place .wav files inside dataset/audio")

if not os.path.exists(metadata_path):
    raise FileNotFoundError("Parkinsons_Dataset.csv not found!")

metadata = pd.read_excel(r"Demographics_age_sex.xlsx")
if "Status" not in metadata.columns or "Sample" not in metadata.columns:
    raise ValueError("CSV must contain 'Sample' and 'Status' columns.")

features = []
labels = []

print("Starting feature extraction...")
for _, row in metadata.iterrows():
    Sample = str(row["Sample"]).strip()
    
    if not Sample:
        print("Skipping row with missing filename.")
        continue 

    file_path = os.path.join(audio_dir, Sample + ".wav")
    
    if not os.path.exists(file_path):
        print(f"Audio file missing: {Sample}")
        continue

    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        avg_mfcc = np.mean(mfcc, axis=1)
        
        features.append(avg_mfcc)
        labels.append(int(row["Status"]))
    except Exception as e:
        print(f"Error processing {Sample}: {e}")

if not features:
    raise ValueError("No audio features extracted. Check file paths and dataset.")

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
pickle.dump(model, open("models/audio_model.sav", "wb"))
pickle.dump(scaler, open("models/audio_scaler.sav", "wb"))

print("Audio model and scaler saved successfully in 'models/' folder.")

example_audio_path = "HC_AH/AH_222K_FC9D2763-1836-460B-954F-37F23D6CD81D.wav"

if os.path.exists(example_audio_path):
    print("\n--- Running Example Prediction ---")
    y, sr = librosa.load(example_audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    avg_mfcc = np.mean(mfcc, axis=1).reshape(1, -1)
    
    input_scaled = scaler.transform(avg_mfcc)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] 

    if prediction[0] == 1:
        print(f"Prediction: Parkinson’s detected with {probability*100:.2f}% confidence.")
    else:
        print(f"Prediction: Healthy speech detected with {(1 - probability)*100:.2f}% confidence.")
else:
    print(f"\nExample audio file not found at: {example_audio_path}")
    print("Skipping example prediction.")