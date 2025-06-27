import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel])

# Load dataset
DATA_PATH = 'audio\Actor_20'
labels = []
features = []

for dirpath, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            label = int(file.split("-")[2])
            path = os.path.join(dirpath, file)
            data = extract_features(path)
            features.append(data)
            labels.append(label)

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

joblib.dump(model, 'emotion_model.pkl')
print("Model trained and saved as emotion_model.pkl")