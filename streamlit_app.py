import logging
import streamlit as st
import gc
import os
import random
import transformers
import warnings

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

    X_test_encoded = encode_text(df=df_test, tokenizer=AutoTokenizer.from_pretrained(
        model_name), max_len=128, padding=True)

    # Build the model
    model = build_model(model_name, 128, "1e-5",
                        ["sparse_categorical_accuracy"])
    model.load_weights("/content/drive/MyDrive/model.h5")

    # , -1, config.AUTO, labelled = False, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)
    ds_test = get_tf_dataset(
        X_test_encoded, auto=tf.data.experimental.AUTOTUNE, labelled=False, y=-1, batch_size=64)
    val = np.argmax(model.predict(ds_test))
    return val


st.cache(show_spinner=False)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "twmkn9/distilbert-base-uncased-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "twmkn9/distilbert-base-uncased-squad2")
    nlp_pipe = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return nlp_pipe


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.ERROR)
st.header("Prototyping an NLP solution")
st.text("This demo uses a model for Question Answering.")
add_text_sidebar = st.sidebar.title("Menu")
add_text_sidebar = st.sidebar.text("Just some random text.")
question = st.text_input(label='Insert a question.')
# text = st.text_area(label="Context")
output = predict_test("bert-large-cased", str(question))
if not (len(str(question)) == 0):
    x_dict = npl_pipe(context=text, question=question
    st.text('Answer: ', output)
