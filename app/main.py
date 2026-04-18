import streamlit as st
from PIL import Image
import tensorflow as tf
from keras_hub.src.models.bert.bert_backbone import BertBackbone
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras_nlp
import re
import pickle
import numpy as np
import pandas as pd

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
    
model = None


def predict(text):
    phrases = clean_for_bert(text)

    probabilities = model.predict(phrases)

    return CLASSES[np.argmax(probabilities, axis=1)[0]]


st.set_page_config(
    page_title="Well beAI",
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
        .result {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Deep Learning", "Machine Learning"])

with tab1:
    st.markdown('<div class="title">Well beAI</div>', unsafe_allow_html=True)

    user_input = st.text_area("Tell me how you feel", height=150)

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
            if model is None:
                model = tf.keras.models.load_model("models/model_half_clean.keras")

            prediction = predict(user_input)

            st.markdown(
                f'<div class="result">{messages[prediction]}</div>',
                unsafe_allow_html=True
            )

pipeline_1 = None
pipeline_2 = None

def encode_lang(x):
    return (x == "fr").astype(int)

def encode_gender(x):
    return (x == "WOM").astype(int)

with tab2:
    st.markdown('<div class="title">Système de recommendation</div>', unsafe_allow_html=True)

    st.write("Répondez aux questions suivantes :")


    # Questions
    USER_GENDER = st.radio("Vous êtes...", ["une femme", "un homme"], horizontal=True)
    USER_AGE = st.number_input("Votre age :", min_value=10, step=1)
    USER_OWNS_MASK = int(st.radio("Avez-vous un masque ?", ["Oui", "Non"], horizontal=True) == "Oui")
    USER_OWNS_SUBSCRIPTION = int(st.radio("Avez-vous un abonnement?", ["Oui", "Non"], horizontal=True) == "Oui")
    USER_APP_LANGUAGE = st.radio("Quelle est votre langue ?", ["français", "english"], horizontal=True)
    USER_USE_ANDROID = int(st.radio("Utilisez-vous ou avez vous utilisé Android sur votre téléphone ?", ["Oui", "Non"], horizontal=True) == "Oui")
    USER_USE_IOS = int(st.radio("Utilisez-vous ou avez vous utilisé iOS sur votre téléphone ?", ["Oui", "Non"], horizontal=True) == "Oui")
    MP_EVITEMENT_SOCRATE = int(st.radio("Lorsque vous devez faire quelque chose qui vous déplaît, en général que faites-vous? ?", ["Je repousse jusqu’à la dernière minute", "Je le fais tout de suite"]) == "Je repousse jusqu’à la dernière minute")
    MP_OPTION_SOCRATE = int(st.radio("Vous avez tendance à ...", ["Faire plusieurs chose en même temps", "Faire une chose à la fois"]) == "Faire plusieurs chose en même temps")

    if st.button("Voir le résultat"):
        v = pd.DataFrame({
            "USER_AGE": [USER_AGE], "USER_GENDER": ["WOM" if USER_GENDER == "une femme" else "MAN"], "USER_APP_LANGUAGE": ["fr" if USER_APP_LANGUAGE == "français" else "en"], "MP_EVITEMENT_SOCRATE": [MP_EVITEMENT_SOCRATE], "MP_OPTION_SOCRATE": [MP_OPTION_SOCRATE], "USER_OWNS_MASK": [USER_OWNS_MASK] ,"USER_OWNS_SUBSCRIPTION": [USER_OWNS_SUBSCRIPTION], "USER_USE_ANDROID": [USER_USE_ANDROID], "USER_USE_IOS": [USER_USE_IOS]})

        if pipeline_1 is None:
            with open("models/1.pkl", "rb") as fn:
                pipeline_1 = pickle.load(fn)

        if pipeline_2 is None:
            with open("models/2.pkl", "rb") as fn:
                pipeline_2 = pickle.load(fn)

        result_1 = pipeline_1.predict(v)
        if result_1[0] == 1:
            st.success("Vous allez aimer le contenu Qu'est-ce que je veux?")
        else:
            st.error("Vous n'allez pas aimer le contenu Qu'est-ce que je veux?")

        result_2 = pipeline_2.predict(v)
        if result_2[0] == 1:
            st.success("Vous allez aimer le programme Apaiser mes pensées")
        else:
            st.error("Vous n'allez pas aimer le programme Apaiser mes pensées")