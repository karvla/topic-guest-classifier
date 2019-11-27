import pickle
import sys
import tensorflow.keras as keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
import sklearn.datasets as skds
from episode import get_labeled
import logging


"""
tfidf and LSTM.
"""

vocab_size = 10000
max_length = 200
batch_size = 20
num_labels = 2
epochs = 15
logging.basicConfig(filename="history.log", level=logging.DEBUG)


def train(train_set):
    episodes = get_labeled(train_set)
    train_samples = [ep.text for ep in episodes]

    X_train = [one_hot(d, vocab_size, lower=False) for d in train_samples]
    X_train = pad_sequences(X_train, maxlen=max_length)

    y_tags = [ep.guest for ep in episodes]
    y_train = to_categorical(y_tags)

    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_length))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    logging.info(model.summary())
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.2,
    )

    return model


def validate(model, val_set):
    episodes = get_labeled(val_set)
    test_samples = [ep.text for ep in episodes]

    X_test = [one_hot(d, vocab_size, lower=False) for d in test_samples]
    X_test = pad_sequences(X_test, maxlen=max_length)

    y_tags = [ep.guest for ep in episodes]
    y_test = to_categorical(y_tags)
    score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

    print("Test score:", score[0])
    print("Test accuracy:", score[1])
    logging.info("Test score:", score[0])
    logging.info("Test accuracy:", score[1])


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

        model = train(labeled_set)
        with open("lstm_model.pickle", "wb") as f:
            pickle.dump(model, f)

        validate(model, test_set)
