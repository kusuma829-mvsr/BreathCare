import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
from audiorecorder import audiorecorder
from pydub import AudioSegment

model = joblib.load("breath_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="BreathCare", layout="centered")

st.title("🫁 BreathCare")
st.subheader("Respiratory Disease Prediction using Voice")


option = st.radio("Select Input Method:", ["🎤 Record Audio", "📂 Upload Audio"])

file_path = None

# RECORD
if option == "🎤 Record Audio":
    audio = audiorecorder("Start Recording", "Stop Recording")

    if len(audio) > 0:
        st.success("Audio recorded successfully ✅")
        st.audio(audio.export().read(), format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio.export(tmp.name, format="wav")
            file_path = tmp.name

# UPLOAD
elif option == "📂 Upload Audio":

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3", "ogg", "flac", "m4a"]
    )

    if uploaded_file is not None:
        st.success("File uploaded successfully ✅")
        st.audio(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            raw_path = tmp.name

        audio = AudioSegment.from_file(raw_path)
        wav_path = raw_path + ".wav"
        audio.export(wav_path, format="wav")

        file_path = wav_path


# PROCESS
if file_path is not None:

    y, sr = librosa.load(file_path, sr=16000)

    duration = len(y) / sr
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    features = np.array([duration, rms, zcr] + mfcc.tolist()).reshape(1, -1)

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    # 🔥 ADD THIS MAPPING (IMPORTANT)
    labels = {
        0: "NORMAL",
        1: "ASTHMA",
        2: "BRONCHITIS",
        3: "PNEUMONIA"
    }

    result = labels.get(int(prediction), "UNKNOWN")

    st.markdown("## 🧠 Prediction Result")

    if result == "NORMAL":
        st.success(f"✅ {result}")
    else:
        st.error(f"⚠️ {result}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Duration (sec)", round(duration, 2))
    col2.metric("RMS Energy", round(float(rms), 5))
    col3.metric("ZCR", round(float(zcr), 5))

    st.markdown("### 📊 MFCC Features")
    mfcc_dict = {f"MFCC_{i+1}": float(val) for i, val in enumerate(mfcc)}
    st.dataframe(mfcc_dict)

else:
    st.info("Please record or upload an audio file")