import pickle
import trie
import sys
from episode import Episode
import matplotlib.pyplot as plt

""" 
Tags parsed episodes with G if NAME is not on wikidata
and with T if name belongs to a fictional character or to a dead person.
And if there aren't to many names in the episode, excluding names only mentioned twice.
"""
def plot_names_histogram(names, title=""):
    n_per_name = list(map(len, names.values()))
    plt.hist(n_per_name, bins=1000, log=True, range=(1, 1200))
    plt.ylabel("Number of names")
    plt.xlabel("Number of mentions")
    plt.title(title)
    plt.savefig("./plots/"+title+ ".png")

def label_set(all_names, dead_names, fict_names):
    """
    Lables the the set.
    """
    topic_names = {}
    guest_names = {}
    for title, description in zip(all_episodes[0::3], all_episodes[1::3]):
        ep = Episode(title, description)
        ep_tokenized = ep.tokenize()

        for tokenized, name in ep_tokenized:

            if name in topic_names:
                topic_names[name].append(tokenized)
                continue

            if name in guest_names:
                guest_names[name].append(tokenized)
                continue

            in_fict, _ = trie.find_prefix(fict_names, name)
            in_dead, _ = trie.find_prefix(dead_names, name)

            if in_fict or in_dead:
                topic_names[name] = [tokenized]
                continue

            in_all, _ = trie.find_prefix(all_names, name)
            if not in_all:
                guest_names[name] = [tokenized]

    
    plot_names_histogram(topic_names, "Names classed as topics")
    plot_names_histogram(guest_names, "Names classed as guests")
    for episode_list in topic_names.values():
        for text in episode_list:
            print(text)
            print("T")
            print()

    for episode_list in guest_names.values():
        n_mentions = len(episode_list)
        if n_mentions > 5 and n_mentions < 25:
            for text in episode_list:
                print(text)
                print("G")
                print()

            


def names_trie(names):
    name_tree = trie.TrieNode('*')
    [trie.add(name_tree, name) for name in names]
    return name_tree

if __name__ == "__main__":
    with open("./data/wikidata_names.txt") as f:
        all_names = names_trie(f.readlines())

    with open("./data/wikidata_names_fict.txt") as f:
        fict_names = names_trie(f.readlines())
        
    with open("./data/wikidata_names_dead.txt") as f:
        dead_names = names_trie(f.readlines())

    file_name = sys.argv[1]
    with open(file_name) as f:
        all_episodes = f.read().splitlines()

    label_set(all_names, dead_names, fict_names)
