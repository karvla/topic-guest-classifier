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
import matplotlib.pyplot as plt

"""
Embedding and LSTM.
"""

vocab_size = 10000
max_length = 100
batch_size = 32
num_labels = 2
epochs = 4
logging.basicConfig(filename="history.log", level=logging.DEBUG)

def xy(labeled_set):
    episodes = [ep for ep in get_labeled(labeled_set)]
    samples = [ep.text for ep in episodes]
    X = [one_hot(d, vocab_size, lower=False) for d in samples]
    X = pad_sequences(X, maxlen=max_length)

    y_tags = [ep.guest for ep in episodes]
    y = to_categorical(y_tags)
    return X, y


def train(train_set, val_set):
    X_train, y_train = xy(train_set)
    X_val, y_val = xy(val_set)

    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_length))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

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
    plt.show()

    return model


def validate(model, val_set):
    X_test, y_test = xy(val_set) 
    score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

    print("Test score:", score[0])
    print("Test accuracy:", score[1])
    logging.info("Test score: " + str(score[0]))
    logging.info("Test accuracy: " + str(score[1]))


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
