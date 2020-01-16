from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot, Tokenizer
import sklearn.metrics as metrics
import pickle
from tabulate import tabulate
from episode import get_labeled, window
import sys
from datetime import date
import numpy as np


def to_percent(n, n_total):
    return str(round(n / n_total * 100, 3)) + "%"


def log_results(y_predicted, y_true, comment=""):

    c = metrics.confusion_matrix(y_true, y_predicted)
    sum_pt = c[0][0] + c[0][1]
    sum_pg = c[1][0] + c[1][1]
    c = [
        ["", "predicted T", "predicted G", ""],
        [
            "true T",
            to_percent(c[0][0], sum_pt),
            to_percent(c[0][1], sum_pt),
            sum_pt,
        ],
        [
            "true G",
            to_percent(c[1][0], sum_pg),
            to_percent(c[1][1], sum_pg),
            sum_pg,
        ],
        ["", c[1][0] + c[0][0], c[1][1] + c[0][1], ""],
    ]

    results = ""
    results += str(date.today()) + ":\n"
    results += comment + "\n\n"
    results += metrics.classification_report(y_true, y_predicted) + "\n"
    results += "Confusion Matrix n=" + str(len(y_true)) + ":\n"
    results += tabulate(c) + "\n"

    print(results)
    with open("./log.txt", "a+") as f:
        f.write(results)


def validate(test_set):
    win_size = 10
    ep_test, _ = get_labeled(test_set, True)
    X_test = [window(ep.description, win_size) for ep in ep_test]
    y_true = [ep.guest for ep in ep_test]

    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=2*win_size+1)

    y_predicted = model.predict(X_test)

    log_results(np.rint(y_predicted), y_true, model_path)


if __name__ == "__main__":

    model_path = sys.argv[1]
    test_set_path = sys.argv[2]

    with open(model_path, "rb") as f:
        model, tokenizer = pickle.load(f)

    with open(test_set_path, "r") as f:
        test_set = f.read()

    validate(test_set)
