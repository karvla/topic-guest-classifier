from episode import Episode
from functools import lru_cache
import regex as re
import spacy
import sys
from termcolor import colored

"""
Builds an unlabled set from all_episodes.txt
"""

nlp = spacy.load("en_core_web_sm")
file_name = sys.argv[1]

with open(file_name) as f:
    all_episodes = f.read().splitlines()

def label_set():
    """
    Lables the the set via user input.
    """
    for title, description in zip(all_episodes[0::3], all_episodes[1::3]):
        ep = Episode(title, description)
        ep_tokenized = ep.tokenize()

        for tokenized, name in ep_tokenized:
            text = title + "\n" + description

            last_pos = 0
            name_len = len(name)
            for it in re.finditer(name, text):
                pos = it.start()
                print(text[last_pos:pos], end="")
                print(colored(name.title(), 'green'), end="")
                last_pos = pos + name_len
            print(text[last_pos:])
            
            print("Is " + colored(name.title(), 'green') + "  a topic (t) or a guest (g)?")
            i = input()
            with open("label_set.txt", "a") as f:
                if i == "t":
                    f.write(tokenized)
                    f.write("\n")
                    f.write("T\n\n")
                elif i == "g":
                    f.write(tokenized)
                    f.write("\n")
                    f.write("G\n\n")
            print()

label_set()


