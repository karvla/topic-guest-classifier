import pickle
import sys
import tensorflow.keras as keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from episode import get_labeled
import logging
import matplotlib.pyplot as plt
from validate import log_results
import regex as re
import numpy as np
from datetime import datetime

"""
Embedding and LSTM.
"""
# Hyper Parameters:
embedding_dim = 300
vocab_size = 10000
max_length = 60
batch_size = 32
num_labels = 1
epochs = 40


def make_embedding_layer(all_texts):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(all_texts)
    sequences = tokenizer.texts_to_sequences(all_texts)
    word_index = tokenizer.word_index

    embeddings_index = {}
    with open("./data/glove.6B.300d.txt") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
            embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    num_words = min(vocab_size, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if word == "focusnamefocus":
            embedding_matrix[i] = np.repeat(0.1, embedding_dim) 
        elif word == "notfocusnot":
            embedding_matrix[i] = np.repeat(-0.1, embedding_dim) 
        elif embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(
        num_words,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_length,
        trainable=False,
    )
    return embedding_layer, tokenizer


def xy(labeled_set, limit=None):
    episodes = [ep for ep in get_labeled(labeled_set, True)]
    samples = [ep.text for ep in episodes]
    X_texts = samples

    y_tags = [ep.guest for ep in episodes]
    return X_texts, y_tags


def train(train_set, val_set):
    X_val, y_val = xy(val_set)
    X_train, y_train = xy(train_set)

    all_texts = []
    all_texts.extend(X_train)
    all_texts.extend(X_val)
    embedding_layer, tokenizer = make_embedding_layer(all_texts)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_length)
    X_val = tokenizer.texts_to_sequences(X_val)
    X_val = pad_sequences(X_val, maxlen=max_length)

    # Model
    model = Sequential()
    model.add(embedding_layer)
    #model.add(Embedding(vocab_size, 8, input_length=max_length))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        weighted_metrics=["accuracy"],
    )
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1)
    estimator = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[es],
    )

    plt.plot(estimator.history["accuracy"])
    plt.plot(estimator.history["val_accuracy"])
    plt.title("Model training")
    plt.ylabel("training acc")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc=0)
    plt.savefig("./plots/" + datetime.now().strftime("%Y-%m-%d_%H:%M") + ".png")

    # Validate
    try:
        comment = sys.argv[2]
    except:
        comment = ""
    y_predicted = np.rint(model.predict(X_val))
    with open("./src/lstm.py") as f:
        text = f.read()
    model_text = re.findall(r"# Model\n([\s\S]+?)\n\n", text, re.M)[0]
    hyperparameter_text = re.findall(r"Hyper Parameters:\n([\s\S]+?)\n\n", text, re.M)[
        0
    ]
    log_results(
        y_predicted, y_val, model_text + "\n" + hyperparameter_text + "\n" + comment
    )

    return model, tokenizer


def validate(model, tokenizer, val_set):
    X_test, y_true = xy(val_set)
    y_predicted = np.rint(model.predict(X_test))

    with open("./src/lstm.py") as f:
        text = f.read()
    model_text = re.findall(r"# Model\n([\s\S]+?)\n\n", text, re.M)[0]
    hyperparameter_text = re.findall(r"Hyper Parameters:\n([\s\S]+?)\n\n", text, re.M)[
        0
    ]
    log_results(y_predicted, y_true, model_text + "\n" + hyperparameter_text)


if __name__ == "__main__":
    file_name = sys.argv[1]

    if file_name[-6:] == "pickle":
        with open(file_name, "rb") as f:
            model = pickle.load(f)
        test_path = sys.argv[2]
        with open(test_path) as f:
            test_set = f.read()
        validate(model, test_set)

    elif file_name[-3:] == "txt":
        with open(file_name) as f:
            labeled_set = f.read()

        with open(file_name[:-9] + "validate.txt") as f:
            test_set = f.read()

        model = train(labeled_set, test_set)
        with open("lstm_model.pickle", "wb") as f:
            pickle.dump(model, f)
