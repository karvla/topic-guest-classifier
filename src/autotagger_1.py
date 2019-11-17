import pickle
import spacy
import trie
import sys
from episode import Episode

with open("./data/names_on_wikipedia.pickle", "rb") as f:
    wiki_names = pickle.load(f)

nlp = spacy.load("en_core_web_sm")
file_name = sys.argv[1]

with open(file_name) as f:
    all_episodes = f.read().splitlines()

def label_set():
    """
    Lables the the set.
    """
    for title, description in zip(all_episodes[0::3], all_episodes[1::3]):
        ep = Episode(title, description)
        ep_tokenized = ep.tokenize()

        for tokenized, name in ep_tokenized:
            in_wiki_names, _ = trie.find_prefix(wiki_names, name)
            if in_wiki_names:
                print(tokenized)
                print("T")
                print()
            else:
                print(tokenized)
                print("G")
                print()

label_set()
