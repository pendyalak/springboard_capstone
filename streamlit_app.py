import logging
import streamlit as st
import gc
import os
import random
import transformers
import warnings
import os.path
from os import path
import requests

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from googletrans import Translator
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModel

st.cache(show_spinner=False)


def encode_text(df, tokenizer, max_len, padding):
    """
    Preprocessing textual data into encoded tokens.
    """
    text = df["text"].values.tolist()

    # encoding text using tokenizer of the model
    text_encoded = tokenizer.batch_encode_plus(
        text,
        pad_to_max_length=padding,
        max_length=max_len
    )

    return text_encoded


st.cache(show_spinner=False)


def get_tf_dataset(X, y, auto, labelled=True, repeat=False, shuffle=False, batch_size=128):
    """
    Creating tf.data.Dataset for TPU.
    """
    if labelled:
        ds = (tf.data.Dataset.from_tensor_slices((X["input_ids"], y)))
    else:
        ds = (tf.data.Dataset.from_tensor_slices(X["input_ids"]))

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(2048)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(auto)

    return ds


st.cache(show_spinner=False)


def build_model(model_name, max_len, learning_rate, metrics):
    """
    Building the Deep Learning architecture
    """
    # defining encoded inputs
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")

    # defining transformer model embeddings
    transformer_model = TFAutoModel.from_pretrained(model_name)
    transformer_embeddings = transformer_model(input_ids)[0]

    # defining output layer
    output_values = Dense(3, activation="softmax")(
        transformer_embeddings[:, 0, :])

    # defining model
    model = Model(inputs=input_ids, outputs=output_values)
    opt = Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = metrics

    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model


st.cache(show_spinner=False)


def predict_test(model_name, test_data):
    """
    Testing the model
    """
    # reading data
    test_dict = {"text": [test_data]}
    df_test = pd.DataFrame(test_dict, columns=["text"])
    print(df_test)
    X_test_encoded = encode_text(df=df_test, tokenizer=AutoTokenizer.from_pretrained(
        model_name), max_len=128, padding=True)

    # Build the model
    model = build_model(model_name, 128, "1e-5",
                        ["sparse_categorical_accuracy"])
    model.load_weights(destination)

    # , -1, config.AUTO, labelled = False, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)
    ds_test = get_tf_dataset(
        X_test_encoded, auto=tf.data.experimental.AUTOTUNE, labelled=False, y=-1, batch_size=64)
    val = np.argmax(model.predict(ds_test))
    return val


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


file_id = '1mIGaPh0npzkCIsNUdyyI-tJbxgtfRvR2'
destination = './model.h5'
if not (path.exists(destination)):
    download_file_from_google_drive(file_id, destination)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.ERROR)
st.header("Spooky Author Identification")
st.text("This demo uses a model for Predicting Author Writings of Edgar Allan Poe, HP Lovecraft, Mary Shelley .")
question = st.text_area(label='Insert a question.')
print("Question:", question)
temp_dict = {0: "Edgar Allan Poe", 1: "HP Lovercraft", 2: "Mary Shelley"}
if not (len(str(question)) == 0):
    print("Question: ", question)
    output = predict_test("bert-large-cased", question)
    print(output)
    out_var = "This belongs to: " + str(temp_dict[output])
    st.write(out_var)
