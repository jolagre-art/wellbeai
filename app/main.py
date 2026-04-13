import streamlit as st
from PIL import Image
import tensorflow as tf
from keras_hub.src.models.bert.bert_backbone import BertBackbone
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras_nlp
import re
import pickle
import numpy as np

CLASSES = ["anxiety", "bipolar", "depression", "normal", "personality disorder", "stress", "suicidal"]

def clean_text(text: str):
    text = text.replace("\n", " ").replace("\"", " ").lower()
    text = re.sub(r"([.,!?'])", r" \1 ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_for_bert(text):
    bert_model_name = "bert_medium_en_uncased"

    bert_phrases = [clean_text(text)]
    preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
        bert_model_name,
        sequence_length=128
    )

    preprocessed_phrases = preprocessor(bert_phrases)
    return preprocessed_phrases


def clean_for_lstm(text):
    max_len = 250

    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        clean_phrase = clean_text(text)
        tokenized_phrases = tokenizer.texts_to_sequences([clean_phrase])
        padded_phrases = pad_sequences(tokenized_phrases, maxlen=max_len, padding='post')
        return padded_phrases
    
model = tf.keras.models.load_model("models/model_half_clean.keras")


def predict(text):
    phrases = clean_for_bert(text)

    probabilities = model.predict(phrases)

    return CLASSES[np.argmax(probabilities, axis=1)[0]]


st.set_page_config(
    page_title="Well bAI",
    page_icon="🫂",
    layout="centered"
)

st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #4A90E2;
        }
        .subtitle {
            text-align: center;
            color: gray;
            margin-bottom: 30px;
        }
        .result {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Well bAI</div>', unsafe_allow_html=True)
#st.markdown('<div class="subtitle">Tell me how you feel</div>', unsafe_allow_html=True)

user_input = st.text_area("Tell me how you feel", height=150)

images = {
    "anxiety": "images/anxiety.png",
    "bipolar": "images/bipolar.png",
    "depression": "images/depression.png",
    "normal": "images/normal.png",
    "personality disorder": "images/personality.png",
    "stress": "images/stress.png",
    "suicidal": "images/suicidal.png",
}

messages = {
    "anxiety": "It seems that you feel anxious 😟",
    "bipolar": "It seems that you may experience mood swings 🎭",
    "depression": "It seems that you feel depressed 💙",
    "normal": "It seems that you feel okay 😊",
    "personality disorder": "It seems there may be personality-related distress 🧩",
    "stress": "It seems that you are stressed 😣",
    "suicidal": "It seems that you may have distressing thoughts 💔",
}

if st.button("Analyze..."):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = predict(user_input)

        st.markdown(
            f'<div class="result">{messages[prediction]}</div>',
            unsafe_allow_html=True
        )

        try:
            img = Image.open(images[prediction])
            st.image(img, width=250)
        except:
            pass
