import pickle
import sys
import tensorflow.keras as keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from episode import get_labeled
import logging
import matplotlib.pyplot as plt
from validate import log_results
import regex as re
import numpy as np

"""
Embedding and LSTM.
"""
#Hyper Parameters:
vocab_size = 10000
max_length = 60
batch_size = 128
num_labels = 1
epochs = 10

def xy(labeled_set, limit=None):
    episodes = [ep for ep in get_labeled(labeled_set)]
    samples = [ep.text for ep in episodes]
    X = [one_hot(d, vocab_size, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n') for d in samples]
    X = pad_sequences(X, maxlen=max_length)

    y_tags = [ep.guest for ep in episodes]
    y = to_categorical(y_tags)
    if limit:
        return X[:limit], y[:limit]

    return X, y_tags

def train(train_set, val_set):
    X_train, y_train = xy(train_set)
    X_val, y_val = xy(val_set)

    # Model
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_length))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    estimator = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, y_val)
    )

    plt.plot(estimator.history['loss'])
    plt.plot(estimator.history['val_loss'])
    plt.title('Model training')
    plt.ylabel('training error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc=0)
    plt.draw()

    return model


def validate(model, val_set):
    X_test, y_true = xy(val_set) 
    y_predicted = np.rint(model.predict(X_test))

    with open("./src/lstm.py") as f:
        text = f.read()
    model_text = re.findall(r"# Model\n([\s\S]+?)\n\n", text, re.M)[0]
    hyperparameter_text = re.findall(r"Hyper Parameters:\n([\s\S]+?)\n\n", text, re.M)[0]
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

        validate(model, test_set)
