import sys
import random
import episode

def episodes_to_txt(episodes):
    text = ""
    for ep in episodes:
        tag = "G" if ep.guest else "T"
        text += ep.text + "\n" + tag + "\n" + "\n" 
    return text

labeled_set_name = sys.argv[1]
with open(labeled_set_name) as f:
    labeled_set = f.read()

samples = episode.get_labeled(labeled_set)
text = ""

n_samples = len(samples)

train_samples = samples[:round(0.8*n_samples)]
val_samples = samples[round(0.8*n_samples):]
test_samples = train_samples[-200:]
train_samples = train_samples[:-200]

train_set = episodes_to_txt(train_samples)
val_set = episodes_to_txt(val_samples)
test_set = episodes_to_txt(test_samples)

with open(labeled_set_name[:-4] + "_train.txt", "w") as f:
    f.write(train_set)

with open(labeled_set_name[:-4] + "_validate.txt", "w") as f:
    f.write(val_set)

with open("test_set.txt", "w") as f:
    f.write(test_set)



