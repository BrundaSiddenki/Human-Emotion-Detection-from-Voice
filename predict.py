import librosa
import numpy as np
import joblib

model = joblib.load("emotion_model.pkl")

EMOTIONS = {
    1: "Neutral",
    2: "Calm",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Fearful",
    7: "Disgust",
    8: "Surprised"
}

EMOJI_MAP = {
    "Neutral": "😐",
    "Calm": "🧘",
    "Happy": "😄",
    "Sad": "😢",
    "Angry": "😠",
    "Fearful": "😨",
    "Disgust": "🤢",
    "Surprised": "😲"
}

def extract_features_from_raw(audio, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel])

def predict_emotion(audio, sr):
    features = extract_features_from_raw(audio, sr).reshape(1, -1)
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    confidence = np.max(probs)
    return EMOTIONS.get(prediction, "Unknown"), EMOJI_MAP.get(EMOTIONS.get(prediction, "Unknown"), ""), confidence
