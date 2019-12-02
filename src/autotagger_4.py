import pickle
import trie
import sys
from episode import Episode
import matplotlib.pyplot as plt

""" 
Tags data but assuming that popular name-mentions will mostly be topics while
not so popular will be guests.
"""
def plot_names_histogram(names, title=""):
    n_per_name = list(map(len, names.values()))
    plt.hist(n_per_name, bins=1000, log=True, range=(1, 1200))
    plt.ylabel("Number of names")
    plt.xlabel("Number of mentions")
    plt.title(title)
    plt.savefig("./plots/"+title+ ".png")

def label_set():
    """
    Lables the the set.
    """
    names = {}
    for title, description in zip(all_episodes[0::3], all_episodes[1::3]):
        ep = Episode(title, description)
        ep_tokenized = ep.tokenize()

        for tokenized, name in ep_tokenized:
            if name in names:
                names[name].append(tokenized)
            else:
                names[name] = [tokenized]
                
    for episode_list in names.values():
        n_mentions = len(episode_list)
        if n_mentions > 400:
            for text in episode_list:
                print(text)
                print("T")
                print()

        elif n_mentions < 20:
            for text in episode_list:
                print(text)
                print("G")
                print()


if __name__ == "__main__":

    file_name = sys.argv[1]
    with open(file_name) as f:
        all_episodes = f.read().splitlines()

    label_set()
