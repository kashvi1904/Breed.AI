import streamlit as st
import tf_keras
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json
import random

st.set_page_config(page_title="Breed.AI 🐾", page_icon="🐶", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;800;900&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Nunito', sans-serif !important;
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 50%, #111 100%) !important;
    color: #f2f2f2 !important;
}

/* HERO */
.hero {
    text-align: center;
    padding: 1.8rem 0 1.3rem;
}
.emoji-row {
    font-size: 2rem;
    letter-spacing: 10px;
    margin-bottom: 0.6rem;
}
.rainbow-title {
    font-size: 3.7rem;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0;
    text-shadow:
        0px 0px 6px rgba(255,255,255,0.25),
        0px 0px 10px rgba(255,255,255,0.15);
    /* crisp glowing white */
}
.sub {
    font-size: 1rem;
    color: #cccccc;
    font-weight: 600;
    margin-top: 6px;
}

/* FILE UPLOADER */
div[data-testid="stFileUploader"] {
    border: 2px dashed #444 !important;
    border-radius: 20px !important;
    background: #1c1c1c !important;
    padding: 1.6rem !important;
    transition: 0.2s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #666 !important;
    background: #222 !important;
}

/* BUTTON */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #3f3fff, #7a3bff) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem 1rem !important;
    font-size: 1.1rem !important;
    font-weight: 900 !important;
    box-shadow: 0 4px 20px rgba(100, 80, 255, 0.25) !important;
    transition: 0.15s ease-in-out;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(100, 80, 255, 0.40) !important;
}

/* RESULT CARD */
.result-card {
    background: #141414;
    border-radius: 22px;
    padding: 2rem;
    text-align: center;
    border: 2px solid #333;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
    margin-top: 1.3rem;
    animation: fadeInUp 0.4s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.woof-emoji {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.result-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #888;
    font-weight: 700;
    margin-bottom: 6px;
}
.breed-name {
    font-size: 2.6rem;
    font-weight: 900;
    color: #8a7fff;
    margin-bottom: 1.2rem;
}

/* CONFIDENCE BAR */
.bar-track {
    height: 12px;
    background: #2a2a2a;
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 10px;
}

/* TAGS */
.tags {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.tag {
    font-size: 0.8rem;
    font-weight: 800;
    padding: 6px 14px;
    border-radius: 99px;
    border: 1px solid #333;
}
.tag-pink { background: #2a1f2e; color: #d87adb; }
.tag-yellow { background: #2a2a1f; color: #d8c56a; }
.tag-blue { background: #1f2633; color: #7ab3ff; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf_keras.models.load_model(
        "model/dog_breed_model.keras",
        custom_objects={"KerasLayer": hub.KerasLayer}
    )

@st.cache_resource
def load_classes():
    with open("model/class_names.json") as f:
        return json.load(f)

model = load_model()
class_names = load_classes()

def preprocess(image):
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

def format_breed(name):
    return name.replace("_", " ").title()

WOOF_EMOJIS = ["🎊", "🥳", "🌟", "✨", "🎉", "🐶", "🦮", "🐕"]

st.markdown("""
<div class="hero">
    <div class="emoji-row">🐶🐾🦴🐕🐾</div>
    <h1 class="rainbow-title">Breed.AI</h1>
    <p class="sub">Which doggo is this? Let's find out! 🎉</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "📸 Drop your doggo photo here!",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_container_width=None)

    if st.button("🔍 Identify this dog!"):
        with st.spinner("🐽 Sniffing out the breed…"):
            arr = preprocess(image)
            preds = model.predict(arr)
            idx = np.argmax(preds[0])
            breed = format_breed(class_names[idx])
            confidence = float(preds[0][idx])
            conf_pct = int(confidence * 100)

        woof = random.choice(WOOF_EMOJIS)
        bar_color = (
            "linear-gradient(90deg, #69db7c, #4dabf7)" if confidence >= 0.8
            else "linear-gradient(90deg, #ffd43b, #ffa94d)" if confidence >= 0.5
            else "linear-gradient(90deg, #ff6b6b, #ffa94d)"
        )
        conf_note = (
            "Super confident! What a distinctive doggo 🌟" if confidence >= 0.8
            else "Pretty sure about this one 🐾" if confidence >= 0.5
            else "Hmm, this one's tricky! 🤔"
        )

        st.markdown(f"""
        <div class="result-card">
            <div class="woof-emoji">{woof}</div>
            <p class="result-label">This dog is a…</p>
            <div class="breed-name">{breed}</div>
            <div class="bar-track">
                <div style="height:100%;border-radius:99px;width:{conf_pct}%;background:{bar_color};"></div>
            </div>
            <p class="conf-text">{conf_pct}% confident — {conf_note}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<p style="text-align:center;font-size:0.75rem;color:#ccc;font-weight:700;margin-top:2rem;">Made with 🧡 · Running on localhost:8501</p>', unsafe_allow_html=True)