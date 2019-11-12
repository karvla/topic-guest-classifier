from episode import Episode
from functools import lru_cache
import regex as re

with open("./all_episodes.txt") as f:
    all_episodes = f.read().splitlines()

names = ["Sam Harris", "Jordan Petersson", "Nick Bostrom", "Hillary Clinton"]

def episodes_with_names(names):
    """
    Takes a list of names and returns a list of episodes where the names occur in 
    the episode description.
    """
    episodes = []
    for name in names:
        for title, description in zip(all_episodes[0::2], all_episodes[1::2]):
            name_in_title = re.findall(name, title)
            name_in_description = re.findall(name, description)
            if name_in_description:
                title = re.sub(name, "NAME", title)
                description = re.sub(name, "NAME", description)
                ep = Episode(title, description)
                episodes.append(ep)
    return episodes


