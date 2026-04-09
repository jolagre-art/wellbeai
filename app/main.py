import streamlit as st
import tensorflow as tf
from keras_hub.src.models.bert.bert_backbone import BertBackbone
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras_nlp
import re
import pickle

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

st.title("Well Be AI")

if prompt := st.chat_input("Tell me something..."):
    phrases = clean_for_bert(prompt)

    probabilities = model.predict(phrases)

    probas = dict()
    for idx in range(0, len(CLASSES)):
        proba = int(probabilities[0][idx] * 10000) / 100.0
        probas[CLASSES[idx]] = proba

    probas = {k: v for k, v in sorted(probas.items(), key=lambda x: x[1], reverse=True) if v > 0}
    for k, v in probas.items():
        st.write(f"{k} {v}")
