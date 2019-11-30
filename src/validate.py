from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics as metrics
import pickle
from tabulate import tabulate
from episode import get_labeled
import sys
from datetime import date

def log_results(y_predicted, y_true, comment=""):

    def to_percent(n):
        return str(round(n/len(y_true)*100, 3))+"%"


    c = metrics.confusion_matrix(y_true, y_predicted)
    c = [["", "actual T", "actual G", ""],
        ["predicted T", c[0][0], c[0][1], c[0][0]+ c[0][1]],
        ["predicted G", c[1][0], c[1][1],c[1][0]+ c[1][1]],
        ["", c[1][0] + c[0][0], c[1][1] + c[0][1], ""]]

    results = ""
    results += str(date.today()) + ":\n"
    results += comment + "\n\n"
    results += metrics.classification_report(y_true, y_predicted) +"\n"
    results += "Confusion Matrix n=" + str(len(y_true)) + ":\n"
    results += tabulate(c) + "\n"

    print(results)
    with open("./log.txt", "a+") as f:
        f.write(results)
        
def validate(model, test_set, corpus, comment="Baseline"):
    episodes = get_labeled(test_set, True)
    corpus_test = [ep.text for ep in episodes]

    vec = TfidfVectorizer()
    vec.fit(corpus)

    X = vec.transform(corpus_test)
    y_true = [ep.guest for ep in episodes]
    y_predicted = model.predict(X)

    log_results(y_predicted, y_true, comment)

if __name__ == "__main__":

    model_path = sys.argv[1]
    test_set_path = sys.argv[2]

    with open(model_path, "rb") as f:
        model, corpus = pickle.load(f)

    with open(test_set_path, "r") as f:
        test_set = f.read()

    validate(model, test_set, tuple(corpus))
