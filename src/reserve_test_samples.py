import sys
import random
import episode

"""
Pickes random 1000 samples without replacement and saves it as a test set.
"""

def episodes_to_txt(episodes):
    text = []
    for ep in episodes:
        text.append(ep.text)
        text.append("\n\n")
    return text

unlabeled_set_name = sys.argv[1]
with open(unlabeled_set_name) as f:
    samples = episode.get_unlabeled(f.read())

random.shuffle(samples)

test_samples = episodes_to_txt(samples[:1000])
samples = episodes_to_txt(samples[1000:])

with open("data/unlabeled_train.txt", "w") as f:
    f.writelines(samples)

with open("data/unlabeled_test.txt", "w") as f:
    f.writelines(test_samples)




