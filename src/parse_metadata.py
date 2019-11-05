import pickle
from pyPodcastParser.Podcast import Podcast
import regex as re
from polyglot.detect import Detector
from langdetect import detect_langs
import requests
import sys

"""
Iterates every RSS feed in feeds_unique.txt and prints the title and
the episode description for every english description.
"""
with open('./feeds_unique.txt') as f:
    rss_feeds = f.readlines()


def _sentences(string):
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

n_total = len(rss_feeds)
for n, feed in enumerate(rss_feeds, 1):
    try:
        response = requests.get(feed, timeout=1.0)
        podcast = Podcast(response.content)
    except:
        continue

    for episode in podcast.items:
        description = episode.description
        if description and _is_english(description):
            sentences = _sentences(description)
            print(episode.title)
            print(sentences)
            print()

