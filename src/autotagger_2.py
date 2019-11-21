import pickle
import trie
import sys
from episode import Episode

""" 
Tags parsed episodes with G if NAME is not on wikidata
and with T if name belongs to a fictional character or to a dead person.
"""


def label_set(all_names, dead_names, fict_names):
    """
    Lables the the set.
    """
    for title, description in zip(all_episodes[0::3], all_episodes[1::3]):
        ep = Episode(title, description)
        ep_tokenized = ep.tokenize()

        for tokenized, name in ep_tokenized:
            in_all, _ = trie.find_prefix(all_names, name)
            in_fict, _ = trie.find_prefix(dead_names, name)
            in_dead, _ = trie.find_prefix(fict_names, name)

            if in_fict or in_dead:
                print(tokenized)
                print("T")
                print()
            elif not in_all:
                print(tokenized)
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
