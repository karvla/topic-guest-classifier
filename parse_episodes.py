import regex as re
import csv
from langdetect import detect_langs
import sys
import pytest

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

def _is_music_mix(string):
    pattern = r"Mix|Original Mix|Remix"
    match = re.findall(pattern, string)
    return None != match

class Test_:
    def test_is_music_mix(self):
        string1 = "This is not a music mix podcast"
        string2 = "This is a music mix podcast feat song (Original Mix)"
        assert _is_music_mix(string1) == False
        assert _is_music_mix(string2) == True

if __name__ == "__main__":

    data_path = sys.argv[1]
    with open(data_path) as f:
        episodes = csv.DictReader(f)
        for row in episodes:
            title = _sentences(row["title"])
            if row["summary"]:
                description = _sentences(row["summary"])
            else:
                description = _sentences(row["description"])
            if title and description and _is_english(description):
                print(title)
                print(description)
                print()
