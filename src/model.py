import pickle
import random
import sys
import tensorflow.keras as keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.utils import to_categorical
from keras import regularizers
from sklearn.preprocessing import LabelBinarizer
from episode import get_labeled, get_unlabeled, window
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
win_size = 30
max_length = win_size*2 + 1
batch_size = 64
num_labels = 1
epochs = 40
bootstrap_set_incr = 0.1
using_bootstrap = False


def make_embedding_layer(all_texts):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(all_texts)
    sequences = tokenizer.texts_to_sequences(all_texts)
    word_index = tokenizer.word_index
    all_texts = None

    with open("./data/embeddings_index.pickle", "rb") as f:
        embeddings_index = pickle.load(f)

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
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(
        num_words,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_length,
        trainable=True,
    )
    return embedding_layer, tokenizer


def make_xy(episodes):
    split_index = int(len(episodes) * 0.8)

    X = [window(ep.description, win_size) for ep in episodes]
    y = [ep.guest for ep in episodes]

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_val = X[split_index:]
    y_val = y[split_index:]
    return X_train, y_train, X_val, y_val


def get_best(ep_labeled, n):
    ep_sorted = sorted(ep_labeled, key=lambda x: x.guest)

    best_T = ep_sorted[:int(n/2)]
    best_G = ep_sorted[-int(n/2):]
    print("Best T range :", best_T[0].guest, " ", best_T[-1].guest)
    print("Best G range :", best_G[0].guest, " ", best_G[-1].guest)
    others = ep_sorted[n:-n]
    episodes = best_G + best_T

    for ep in episodes:
        ep.guest = int(ep.guest)

    return episodes, others


def bootstrap(ep_train, ep_unlabeled):

    def n_new():
        return int(len(ep_train)*bootstrap_set_incr)

    while len(ep_unlabeled) - n_new() > 0:
        model, accuracy, tokenizer = train(ep_train)
        ep_predicted = predict(ep_unlabeled, model, tokenizer)
        ep_best, ep_unlabeled = get_best(ep_predicted, n_new())
        ep_train.extend(ep_best)
        random.shuffle(ep_train)
        print("Accuracy: ", accuracy)
        print()

    return model, tokenizer


def train(ep_train):
    X_train, y_train, X_val, y_val = make_xy(ep_train)

    embedding_layer, tokenizer = make_embedding_layer(X_val + X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_length)
    X_val = tokenizer.texts_to_sequences(X_val)
    X_val = pad_sequences(X_val, maxlen=max_length)

    # Model
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(32, kernel_regularizer=regularizers.l2(0.01))))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=1)

    estimator = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[es],
    )

    accuracy = estimator.history["val_accuracy"][-1]

    #plt.plot(estimator.history["accuracy"])
    plt.plot(estimator.history["val_accuracy"])
    plt.title("Model training")
    plt.ylabel("training acc")
    plt.xlabel("epoch")
    plt.legend(["validation"], loc=0)
    plt.savefig("./plots/" + datetime.now().strftime("%Y-%m-%d_%H:%M") + ".png")

    y_predicted = np.rint(model.predict(X_val))

    log_results(y_predicted, y_val, comment)

    return model, accuracy, tokenizer


def predict(ep_unlabeled, model, tokenizer):
    X = [ep.text for ep in ep_unlabeled]
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=max_length)
    y_predicted = model.predict(X)

    for ep, y in zip(ep_unlabeled, y_predicted):
        ep.guest = y
    ep_predicted = ep_unlabeled
    return ep_predicted


if __name__ == "__main__":
    labeled_name = sys.argv[1]
    if len(sys.argv) > 2:
        comment = sys.argv[2]
    else:
        comment = ""

    with open(labeled_name) as f:
        labeled_set = f.read()

    ep_train, ep_unlabeled = get_labeled(labeled_set, True)

    if using_bootstrap:
        model = bootstrap(ep_train, ep_unlabeled)
    else:
        model, accuracy, tokenizer = train(ep_train)
        model = (model, tokenizer)

    with open("lstm_model.pickle", "wb") as f:
        pickle.dump(model, f)
