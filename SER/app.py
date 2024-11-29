import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from io import BytesIO
model = tf.keras.models.load_model('best_model.keras')
def predict_emotion(audio_segment):
    audio_segment = np.expand_dims(audio_segment, axis=0)  
    prediction = model.predict(audio_segment)
    emotion_index = np.argmax(prediction)
    emotions = ['Wonder', 'Love', 'Empathy', 'Anger', 'Fear', 'Trust', 'Nostalgia']  
    return emotions[emotion_index]
def normalize(wav):
    rms = np.sqrt(np.mean(np.square(wav)))
    if rms > 0:
        return 0.1 * wav / rms
    else:
        return wav

def process_audio(audio_bytes, seg_len=16000, seg_ov=0.5):
    audio_data, sr = librosa.load(BytesIO(audio_bytes), sr=16000)
    audio_data = normalize(audio_data)

    if len(audio_data) < seg_len:
        audio_data = np.pad(audio_data, (0, seg_len - len(audio_data)))

    segments = []
    step = int(seg_len * (1 - seg_ov))
    for start in range(0, len(audio_data) - seg_len + 1, step):
        segments.append(audio_data[start:start + seg_len])

    return segments
st.set_page_config(page_title="Emotion Detection", page_icon="🎙️", layout="centered")
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .main-container {
        max-width: 700px;
        margin: auto;
        padding: 20px;
        border-radius: 10px;
        background: #1c2732;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.3);
    }
    h1, h3 {
        text-align: center;
    }
    .segment-card {
        background: #ffffff;
        color: #000000;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.2);
    }
    .emotion {
        color: #4CAF50;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<h1>🎙️ Emotion Detection from Audio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an audio file (wav/mp3) to analyze its emotional content.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav', start_time=0)
    st.markdown("<h3>Processing Audio...</h3>", unsafe_allow_html=True)

    with st.spinner("🔄 Analyzing emotions..."):
        segments = process_audio(uploaded_file.read())

    if segments:
        st.success("🎉 Processing complete!")
        st.markdown("<h3>Predicted Emotions:</h3>", unsafe_allow_html=True)

        for i, segment in enumerate(segments):
            emotion = predict_emotion(segment)
            st.markdown(
                f"""
                <div class='segment-card'>
                    <h4>Segment {i + 1}</h4>
                    <p>Predicted Emotion: <span class='emotion'>{emotion}</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.error("⚠️ Error: Could not process the audio file. Please try again.")
else:
    st.info("📂 Please upload an audio file to start analyzing emotions.")

st.markdown("</div>", unsafe_allow_html=True)
