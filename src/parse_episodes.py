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
    string = re.sub(r"http\S+", "", string)
    string = re.sub(r"\s+", " ", string)
    string = re.sub(r"\n", "", string)
    string = re.sub(r"\s+", " ", string)
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
    return [] != match

def parse(data_path):
    with open(data_path, "r") as f:
        episodes = csv.DictReader(f)

        for row in episodes:
            title = _sentences(row["title"])
            if row["summary"]:
                description = _sentences(row["summary"])
            else:
                description = _sentences(row["description"])
            if (
                title
                and description
                and _is_english(description)
                and not _is_music_mix(description)
            ):
                print(title)
                print(description)
                print()

if __name__ == "__main__":
    data_path = sys.argv[1]
    parse(data_path)
