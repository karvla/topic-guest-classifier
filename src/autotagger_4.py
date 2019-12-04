import pickle
import trie
import sys
from episode import Episode
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle


""" 
Tags data but assuming that popular name-mentions will mostly be topics while
not so popular will be guests.
"""
def plot_names_histogram(names, limits=(20, 400)):

    n_per_name = list(map(len, names.values()))

    plt.figure(figsize=(5,4))
    N, bins, patches = plt.hist(n_per_name, bins=60, log=True, range=(0, 600), color='grey')
    colors = ['#EA5739','#6de581']

    print(len(patches))
    for i in range(0, 2):
        patches[i].set_color(colors[1])
        patches[i].set_edgecolor('black')
        patches[i].set_hatch('//')

    for i in range(40, 60):
        patches[i].set_color(colors[0])
        patches[i].set_edgecolor('black')
        patches[i].set_hatch('*')

    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

    circ1 = mpatches.Patch( facecolor=colors[1],hatch='//',label='Guest Names')
    circ2= mpatches.Patch( facecolor=colors[0],hatch='*',label='Topic Names')
    plt.legend(handles = [circ1,circ2], frameon=False)

    plt.ylabel("Unique Name Count", fontdict=font)
    plt.xlabel("Number of occurrences in the corpus", fontdict=font)
    plt.title("Distribution of Unique Name Occurrences", fontdict=font)

    plt.savefig("./plots/hist.svg")
    plt.show()
    plt.draw()

def get_name_dict():
    names = {}
    for title, description in zip(all_episodes[0::3], all_episodes[1::3]):
        ep = Episode(title, description)
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

    plot_names_histogram(names)
                
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
