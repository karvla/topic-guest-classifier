from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from episode import get_labeled
import sys


def validate(model, test_set, corpus):
    episodes = get_labeled(test_set)
    corpus_test = [ep.text for ep in episodes]

    vec = TfidfVectorizer()
    vec.fit(corpus)

    X = vec.transform(corpus_test)
    y = [ep.guest for ep in episodes]
    y_predicted = model.predict(X)

    n_total = len(y)
    n_correct = 0
    for i, j in zip(y, y_predicted):
        if i == j:
            n_correct += 1

    print("Accurrycy: " + str(n_correct / n_total * 100) + "%")


if __name__ == "__main__":

    model_path = sys.argv[1]
    test_set_path = sys.argv[2]

    with open(model_path, "rb") as f:
        model, corpus = pickle.load(f)

    with open(test_set_path, "r") as f:
        test_set = f.read()

    validate(model, test_set, corpus)
