import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from episode import get_labeled
import sys 
import validate

"""
tf-idf and logistic regression.
"""

def train(train_set):
    episodes = get_labeled(train_set)
    corpus = [ep.text for ep in episodes]
    vec = TfidfVectorizer()
    
    print("Vectorizing..")
    X = vec.fit_transform(corpus)
    y = [ep.guest for ep in episodes]
    
    classifier = linear_model.LogisticRegression(
        penalty="l2", dual=True, solver="liblinear"
    )
    model = classifier.fit(X, y)
    print("Confusion matrix:")
    print(model)
    return model, corpus


if __name__ == "__main__":
    file_name = sys.argv[1]

    with open(file_name) as f:
        labeled_set = f.read()

    model, corpus = train(labeled_set)

    with open("baseline_model.pickle", "wb") as f:
        pickle.dump((model, corpus), f)

