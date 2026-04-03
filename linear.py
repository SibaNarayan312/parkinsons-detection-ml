import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import os

metadata_path = "Parkinsons_Dataset.csv"
if not os.path.exists(metadata_path):
    raise FileNotFoundError("Dataset not found! Please place parkinsons.csv inside the dataset folder.")

parkinsons_data = pd.read_csv(r"Parkinsons_Dataset.csv")

X = parkinsons_data.drop(columns=['name','status'])
Y = parkinsons_data['status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))
print(f"Training Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

os.makedirs("models", exist_ok=True)
filename = 'models/parkinsons_model.sav'
pickle.dump(model, open(filename, 'wb'))
print(f"Model saved as {filename}")

input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,
0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)
input_data_np = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_data_np)

if prediction[0] == 0:
    print("The Person does NOT have Parkinson’s Disease.")
else:
    print("The Person HAS Parkinson’s Disease.")