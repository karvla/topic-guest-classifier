import regex as re
import csv
from langdetect import detect_langs

"""
Parses episodes.csv.
"""



def _sentences(string):
    string = re.sub("<a href=", "", string)
    string = re.sub("<.*?>", "", string)
    string = re.sub("http\S+", "", string)
    string = re.sub("\s+", " ", string)
    string = re.sub("\n", "", string)
    string = re.sub("\s+", " ", string)
    return string


def _is_english(string):
    try:
        lang = detect_langs(string)
    except:
        return False
    return lang[0].lang == "en" and lang[0].prob > 0.95

with open('../data/episodes.csv') as f:
    episodes = csv.DictReader(f)
    for row in episodes:
        title = row["title"]
        if row["summary"]:
            description = _sentences(row["summary"])
        else:
            description = _sentences(row["description"])
        if title and description and _is_english(description):
            print(title)
            print(description)
            print("_")
            print()




