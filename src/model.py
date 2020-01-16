import pickle
import random
import sys
import tensorflow.keras as keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.utils import to_categorical
from keras import regularizers
from episode import get_labeled, get_unlabeled, window
import logging
import matplotlib.pyplot as plt
from validate import log_results
import regex as re
import numpy as np
from datetime import datetime
from guppy import hpy; h=hpy()
import keras.backend as K

"""
Used for training the model. The model uses bidir-LSTM and GloVe word embedding.
"""

# Hyper Parameters:
embedding_dim = 50
vocab_size = 5000
win_size = 10
max_length = win_size * 2 + 1
batch_size = 64
num_labels = 1
epochs = 40
bootstrap_set_incr = 0.1
using_bootstrap = False
model_name = "./models/lstm_name_dist.pickle"


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
            embedding_matrix[i] = np.repeat(0.5, embedding_dim)
        elif word == "notfocusnot":
            embedding_matrix[i] = np.repeat(-0.5, embedding_dim)
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


def down_sample(a, b):
    max_index = min((len(a), len(b))) - 1
    return a[:max_index], b[:max_index]


def make_xy(episodes):
    random.shuffle(episodes)
    split_index = int(len(episodes) * 0.8)

    X = [window(ep.description, win_size) for ep in episodes]
    # X = [window(ep.description, win_size) for ep in episodes]
    # X = [ep.focus_sentence() for ep in episodes]
    y = [ep.guest for ep in episodes]
    episodes = None

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_val = X[split_index:]
    y_val = y[split_index:]
    return X_train, y_train, X_val, y_val


def bootstrap(ep_train_path, ep_unlabeled_path):

    with open(ep_train_path) as f:
        ep_train = np.array(get_labeled(f.read(), False))[0]

    with open(ep_unlabeled_path) as f:
        ep_unlabeled = np.array(get_unlabeled(f.read(), 4))

    def n_new():
        return int(len(ep_train) * bootstrap_set_incr)

    phase = 0
    model_name = "./models/semi_supervised_non_best_phase_"
    while len(ep_unlabeled) - n_new() > 0:
        print("Phase ", phase)
        print(h.heap())

        model, accuracy, tokenizer = train(ep_train)

        print("Predicting and selecting..")
        ep_best, ep_unlabeled = get_best(ep_unlabeled, model, tokenizer, n_new())
        ep_train = np.concatenate((ep_train, ep_best), axis=None)
        ep_best = None
        random.shuffle(ep_train)
        K.clear_session()

        if phase % 5 == 0:
            with open(model_name + str(phase) + ".pickle", "wb") as f:
                pickle.dump((model, tokenizer), f)
        phase += 1

    return model, tokenizer


def train(ep_train):
    X_train, y_train, X_val, y_val = make_xy(ep_train)
    ep_train = None

    embedding_layer, tokenizer = make_embedding_layer(X_val + X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_length)
    X_val = tokenizer.texts_to_sequences(X_val)
    X_val = pad_sequences(X_val, maxlen=max_length)

    # Model
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.2)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)

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

    # plt.plot(estimator.history["accuracy"])
    plt.plot(estimator.history["val_accuracy"])
    plt.title("Model training")
    plt.ylabel("training acc")
    plt.xlabel("epoch")
    plt.legend(["validation"], loc=0)
    plt.savefig("./plots/" + datetime.now().strftime("%Y-%m-%d_%H:%M") + ".png")

    y_predicted = np.rint(model.predict(X_val))

    log_results(y_predicted, y_val, comment)

    return model, accuracy, tokenizer

def get_best(episodes, model, tokenizer, n):
    """
    Predicts unlabeled sampels and retrus the best ones
    """
    X = [window(ep.description, win_size) for ep in episodes]
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=max_length)
    y_predicted = model.predict(X)
    X = None

    for ep, y in zip(episodes, y_predicted):
        ep.guest = float(y[0])

    #episodes = np.array(sorted(episodes, key=lambda x: x.guest))
    best = []
    non_best = []
    n_T = 0
    n_G = 0
    for ep in episodes:
        if ep.guest > 0.95 and n_G <= n:
            n_G += 1
            ep.guest = 1
            best.append(ep)
        elif ep.guest < 0.05 and n_T <= n:
            n_T += 1
            ep.guest = 0
            best.append(ep)
        else:
            non_best.append(ep)
    episodes = best

    #non_best = episodes[n:-n]
    #episodes = np.concatenate((episodes[:n], episodes[-n:]), axis=None)
    random.shuffle(episodes)
    random.shuffle(non_best)
    
    #for ep in episodes:
    #    ep.guest = round(ep.guest)

    return episodes, non_best


if __name__ == "__main__":
    labeled_name = sys.argv[1]
    if len(sys.argv) > 2:
        comment = sys.argv[2]
    else:
        comment = ""

    #with open(labeled_name) as f:
    #    ep = get_labeled(f.read())

    with open(labeled_name) as f:
        ep_train = np.array(get_labeled(f.read(), True)[0])

    #ep_unlabeled = np.array(ep[0] + ep[1])
    #ep = None

    if using_bootstrap:
        model = bootstrap("./data/hand_annotated_train.txt", labeled_name)
    else:
        model, accuracy, tokenizer = train(ep_train)
        model = (model, tokenizer)

    with open(model_name, "wb") as f:
       pickle.dump(model, f)
