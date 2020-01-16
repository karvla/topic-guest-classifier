import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from episode import get_labeled, window
import sys 
from validate import log_results
from model import make_xy

"""
The baseline model uses tf-idf and logistic regression.
"""

def train_and_validate(ep_train, ep_test=None):
    X_train = [window(ep.description, 10) for ep in ep_train]
    y_train = [ep.guest for ep in ep_train]
    X_test = [window(ep.description, 10) for ep in ep_test]
    y_test = [ep.guest for ep in ep_test]

    print("Fitting on corpus")
    vec = TfidfVectorizer()
    vec.fit(X_train)

    print("Transfroming")
    X_train = vec.transform(X_train)
    X_test = vec.transform(X_test)
    classifier = linear_model.LogisticRegression(
        penalty="l2", dual=True, solver="liblinear"
    )
    print("Fitting model")
    model = classifier.fit(X_train, y_train)

    print("Predicting")
    y_predicted = model.predict(X_test)

    log_results(y_predicted, y_test, comment)




if __name__ == "__main__":
    file_name_train = sys.argv[1]
    file_name_test = sys.argv[2]
    comment = sys.argv[3]

    with open(file_name_train) as f:
        train_set = f.read()

    with open(file_name_test) as f:
        test_set = f.read()

    ep_train, _ = get_labeled(train_set, True)
    ep_test, _ = get_labeled(test_set, True)
    train_and_validate(ep_train, ep_test)

