import pickle
import sys
from episode import Episode, get_unlabeled


""" 
Tags data but assuming that popular name-mentions will mostly be topics while
not so popular will be guests.
"""
g_lim = (0, 20)
t_lim = (400, 10000)

def get_name_dict():
    names = {}
    for ep in get_unlabeled(all_episodes):
        ep_tokenized = ep.tokenize()

        for tokenized, name in ep_tokenized:
            if name in names:
                names[name].append(tokenized)
            else:
                names[name] = [tokenized]
    
    return names


def label_set():
    """
    Lables the the set.
    """
    names = get_name_dict()
    with open('./names.pickle', 'wb') as f:
        pickle.dump(names, f)

    for episode_list in names.values():
        n_mentions = len(episode_list)
        if n_mentions > t_lim[0] and n_mentions < t_lim[1]:
            for text in episode_list:
                print(text)
                print("T")
                print()

        elif n_mentions > g_lim[0] and n_mentions < g_lim[1]:
            for text in episode_list:
                print(text)
                print("G")
                print()

        else:
            for text in episode_list:
                print(text)
                print("_")
                print()


if __name__ == "__main__":

    file_name = sys.argv[1]
    with open(file_name) as f:
        all_episodes = f.read()

    label_set()
