import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import sys

def print_top_names(names):
    name_len = [(name, len(names[name])) for name in names.keys()]
    print(sorted(name_len, key=lambda x: x[1]))


def plot_names_histogram(names, limits=(20, 400)):

    n_per_name = list(map(len, names.values()))

    n_bins = 30

    plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    N, bins, patches = plt.hist(n_per_name, bins=n_bins, log=True, range=(0, 600), color='grey')
    colors = ['#EA5739','#6de581']
    hatches = ['///', '\\\\']

    for i in range(0, 1):
        patches[i].set_color(colors[1])
        patches[i].set_edgecolor('black')
        patches[i].set_hatch(hatches[1])

    for i in range(20, 30):
        patches[i].set_color(colors[0])
        patches[i].set_edgecolor('black')
        patches[i].set_hatch(hatches[0])

    font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 14,
        }

    circ1 = mpatches.Patch(facecolor=colors[1],hatch=hatches[1],label='Guest Names')
    circ2= mpatches.Patch(facecolor=colors[0],hatch=hatches[0],label='Topic Names')
    plt.legend(handles = [circ1,circ2], frameon=False, prop=font)

    plt.ylabel("Unique Name Count", fontdict=font)
    plt.xlabel("Number of occurrences in the corpus", fontdict=font)
    plt.title("Distribution of Unique Name Occurrences", fontdict=font)

    plt.savefig("./plots/hist.svg")
    plt.show()

if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        data = pickle.load(f)

    print_top_names(data)
    plot_names_histogram(data)
