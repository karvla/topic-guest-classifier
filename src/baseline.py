import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from episode import get_labeled
import sys 
from validate import log_results

"""
tf-idf and logistic regression.
"""

def train_and_validate(train_set, val_set):
    train_episodes = get_labeled(train_set, True)
    val_episodes = get_labeled(val_set, True)

    all_ep = []
    all_ep.extend(train_episodes)
    all_ep.extend(val_episodes)

    corpus = [ep.text for ep in all_ep]
    corpus_train = [ep.text for ep in train_episodes]
    corpus_val = [ep.text for ep in val_episodes]

    print("Fitting on corpus")
    vec = TfidfVectorizer()
    vec.fit(corpus)

    X_val = vec.transform(corpus_val)
    X_train = vec.transform(corpus_train)
    y_val = [ep.guest for ep in val_episodes]
    y_train = [ep.guest for ep in train_episodes]

    print("Transfroming")
    X = vec.transform(corpus)
    classifier = linear_model.LogisticRegression(
        penalty="l2", dual=True, solver="liblinear"
    )
    print("Fitting model")
    model = classifier.fit(X_train, y_train)

    print("Predicting")
    y_predicted = model.predict(X_val)
    y_true = y_val

    log_results(y_predicted, y_true, comment)

if __name__ == "__main__":
    file_name_train = sys.argv[1]
    file_name_val = sys.argv[2]
    comment = sys.argv[3]

    with open(file_name_train) as f:
        train_set = f.read()

    with open(file_name_val) as f:
        val_set = f.read()

    train_and_validate(train_set, val_set)

