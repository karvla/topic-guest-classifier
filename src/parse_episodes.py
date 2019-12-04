import regex as re
import csv
from langdetect import detect_langs
import sys
import pytest
from pyPodcastParser.Podcast import Podcast
import requests

"""
Parses episodes.csv.
"""


def _sentences(string):
    try:
        string = re.sub("<a href=", "", string)
        string = re.sub("<.*?>", "", string)
        string = re.sub(r"http\S+", "", string)
        string = re.sub(r"\s+", " ", string)
        string = re.sub(r"\n", "", string)
        string = re.sub(r"\s+", " ", string)
    except:
        pass
    return string


def _is_english(string):
    try:
        lang = detect_langs(string)
    except:
        return False
    return lang[0].lang == "en" and lang[0].prob > 0.95


def print_episode(title, description):
    if (
        title
        and description
        and _is_english(description)
        and not _is_music_mix(description)
    ):
        print(title)
        print(description)
        print()


def _is_music_mix(string):
    pattern = r"Mix|Original Mix|Remix"
    match = re.findall(pattern, string)
    return [] != match


def download_metadata(data_path):
    with open(data_path, "r") as f:
        shows = csv.DictReader(f)
        for row in shows:
            url = row["feed_url"]
            try:
                response = requests.get(url, timeout=1.0)
                podcast = Podcast(response.content)
            except:
                continue
            for episode in podcast.items:
                title = _sentences(episode.title)
                if episode.description:
                    description = _sentences(episode.description)
                    print_episode(title, description)


def parse_episodes(data_path):
    with open(data_path, "r") as f:
        episodes = csv.DictReader(f)

        for row in episodes:
            title = _sentences(row["title"])
            if row["summary"]:
                description = _sentences(row["summary"])
            else:
                description = _sentences(row["description"])
            print_episode(title, description)


if __name__ == "__main__":
    data_path = sys.argv[1]
    if len(sys.argv) > 2:
        download_metadata(data_path)
    else:
        parse(data_path)
