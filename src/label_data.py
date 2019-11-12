from episode import Episode
from functools import lru_cache
import regex as re
import spacy


"""
Builds an unlabled set from all_episodes.txt
"""

nlp = spacy.load("en_core_web_sm")
file_name = "label_data.txt"

with open("./all_episodes.txt") as f:
    all_episodes = f.read().splitlines()

def label_set():
    """
    Lables the the set via user input.
    """
    for title, description in zip(all_episodes[0::2], all_episodes[1::2]):
        ep = Episode(title, description)
        episodes_tokenized = ep.tokenize()
        #TODO: Remove already labeled data
        #with open("./all_episodes.txt", "w") as f:
        #    f.writelines(all_episodes[3:])

        for ep_tokenized in episodes_tokenized:
            print(ep_tokenized)
            i = input("Is NAME a topic (t) or a guest (g)?")
            with open("label_set.txt", "a") as f:
                if i == "t":
                    f.write(ep_tokenized)
                    f.write("\n")
                    f.write("T\n\n")
                elif i == "g":
                    f.write(ep_tokenized)
                    f.write("\n")
                    f.write("G\n\n")
            print()

label_set()


