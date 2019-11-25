import pickle
import sys
import tensorflow.keras as keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
import sklearn.datasets as skds

from pathlib import Path
from episode import get_labeled

"""
tfidf and ANN.
"""

vocab_size = 5000
batch_size = 20 
num_labels = 2
epochs = 15

def train(train_set):
    episodes = get_labeled(train_set)
    train_samples = [ep.text for ep in episodes]

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train_samples)
    X_train = tokenizer.texts_to_matrix(train_samples, mode='tfidf')

    y_tags = [ep.guest for ep in episodes]
    y_train = to_categorical(y_tags)

    model = Sequential()
    model.add(Dense(512, input_shape=(vocab_size,)))
    model.add(Activation('relu'))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)


    return model 

def validate(model, val_set):
    episodes = get_labeled(val_set)
    test_samples = [ep.text for ep in episodes]

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(test_samples)


    X_test = tokenizer.texts_to_matrix(test_samples, mode='count')

    y_tags = [ep.guest for ep in episodes]
    y_test = to_categorical(y_tags)
    score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    



if __name__ == "__main__":
    file_name = sys.argv[1]

    if file_name[-6:] == "pickle":
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
        test_path =  sys.argv[2]
        with open(test_path) as f:
            test_set = f.read()
        validate(model, test_set)

    elif file_name[-3:] == "txt":
        with open(file_name) as f:
            labeled_set = f.read()

        with open(file_name[:-9] + "validate.txt") as f:
            test_set = f.read()

        model = train(labeled_set)
        with open("bag_of_words_model.pickle", "wb") as f:
            pickle.dump(model, f)

        validate(model, test_set)

