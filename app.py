# app.py
import streamlit as st
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import speech_recognition
from langdetect import detect
from collections import Counter
from predict import predict_emotion

# --- Page Config ---
st.set_page_config(page_title="🎤 Voice Emotion AI", layout="centered")

# --- Session Initialization ---
if "app_started" not in st.session_state:
    st.session_state.app_started = False
if 'history' not in st.session_state:
    st.session_state.history = []
    st.session_state.emotions = []
    st.session_state.confidences = []

# --- Light CSS Theme (white background) ---
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #333333;
}
.stButton > button {
    background-color: #007bff;
    color: white;
    padding: 0.6rem 1.5rem;
    font-size: 1rem;
    border-radius: 8px;
    border: none;
}
.stButton > button:hover {
    background-color: #3399ff;
}
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

# --- Splash Page ---
if not st.session_state.app_started:
    st.image("https://cdn-icons-png.flaticon.com/512/6816/6816146.png", width=140)
    st.markdown("<h1 style='text-align:center;'>🎧 Welcome to Voice Emotion AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>“Let your voice tell your story...”</p>", unsafe_allow_html=True)
    if st.button("🚀 Launch Emotion App"):
        st.session_state.app_started = True
    st.stop()

# --- Sidebar Navigation ---
page = st.sidebar.radio("📁 Navigation", ["🎙️ Emotion Detector", "📘 About", "🆘 Help"])

# --- Language Mapper ---
def get_language_name(code):
    mapping = {
        "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
        "kn": "Kannada", "ml": "Malayalam", "fr": "French", "de": "German",
        "es": "Spanish", "zh-cn": "Chinese"
    }
    return mapping.get(code.lower(), "Unknown")

# --- Transcription ---
def transcribe_audio(audio, sr):
    recognizer = speech_recognition.Recognizer()
    try:
        sf.write("temp.wav", audio, sr)
        with speech_recognition.AudioFile("temp.wav") as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            lang_code = detect(text)
            return text, get_language_name(lang_code)
    except:
        return "Could not transcribe.", "Unknown"

# --- Logging ---
def save_log(source, emotion):
    log = pd.DataFrame([[source, emotion]], columns=['Source', 'Emotion'])
    log.to_csv("prediction_log.csv", mode='a', header=not os.path.exists("prediction_log.csv"), index=False)

# --- Graphs ---
def show_graph():
    if st.session_state.emotions:
        fig, ax = plt.subplots(1, 2, figsize=(18, 5))
        ax[0].plot(st.session_state.emotions, marker='o', linestyle='-', color='purple')
        ax[0].set_title("Emotion Trend")
        ax[0].set_xticks(range(len(st.session_state.history)))
        ax[0].set_xticklabels(st.session_state.history, rotation=45)
        count = Counter(st.session_state.emotions)
        ax[1].bar(count.keys(), count.values(), color='orange')
        ax[1].set_title("Emotion Distribution")
        st.pyplot(fig)

# --- Main Page: Emotion Detection ---
if page == "🎙️ Emotion Detector":
    st.header("🎧 Voice Emotion Detector")
    audio_files = st.file_uploader("📤 Upload .wav files", type=['wav'], accept_multiple_files=True)

    if audio_files:
        for audio_file in audio_files:
            st.subheader(f"📁 {audio_file.name}")
            audio_data, sr = sf.read(audio_file)
            if audio_data.size == 0:
                st.error("❌ Audio file is empty or invalid.")
            else:
                st.audio(audio_file)
                emotion, emoji, confidence = predict_emotion(audio_data, sr)
                subtitle, language = transcribe_audio(audio_data, sr)
                st.success(f"{emoji} Emotion: **{emotion}**  | Confidence: `{confidence:.2f}`")
                st.info(f"💬 Transcription: _{subtitle}_")
                st.warning(f"🌍 Language Detected: `{language}`")
                st.session_state.history.append(audio_file.name)
                st.session_state.emotions.append(emotion)
                st.session_state.confidences.append(confidence)
                save_log(audio_file.name, emotion)

        show_graph()

    if os.path.exists("prediction_log.csv"):
        with open("prediction_log.csv", "rb") as f:
            st.download_button("📥 Download Emotion Log", f, file_name="emotion_log.csv")

# --- About Page ---
elif page == "📘 About":
    st.header("📘 About")
    st.markdown("""
**Voice Emotion AI** is a smart emotion recognition system that analyzes human speech.  
Features include:
- 🎧 Audio Feature Extraction  
- 🤖 ML-based Emotion Detection  
- 💬 Speech-to-Text  
- 🌍 Auto Language Detection
    """)

# --- Help Page ---
elif page == "🆘 Help":
    st.header("🆘 Help")
    st.markdown("""
**How to Use:**
1. Upload a `.wav` file  
2. App shows:  
   - 🎭 Predicted emotion  
   - 💬 Transcription  
   - 🌍 Language  
3. 📊 View emotion graphs  
4. 📥 Export results  
    """)

