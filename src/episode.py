import nltk
from nltk.tag.stanford import StanfordNERTagger
import regex as re

st = StanfordNERTagger('/home/karvla/projects/topic-guest-classifier/english.all.3class.distsim.crf.ser.gz', '/home/karvla/projects/topic-guest-classifier/stanford-ner.jar')


class Episode:
    def __init__(self, title, description, guest=False):
        self.title = title
        self.description = description
        self.guest = guest
        self.text = title + '\n' + description

    def print(self):
        print(self.title)
        print(self.description)

    def window(self, win_size=3):
        padding = ["NIL" for n in range(win_size)]
        words = padding + self.description.split() + padding
        window = None
        for i, word in enumerate(words):
            if word == "NAME":
                window = []
                window.extend(words[i - win_size : i])
                window.extend(words[i : i + win_size + 1])
        return window

    def tokenize(self):
        """ 
        Returns a list of tokenized titles and descriptions, names names and pos,
        where the names are replaced with "NAME":
        """
        texts = []
        names = self.names()

        for name in names:
            pattern = r"<>" + name + "<\\\>"
            tok_text = re.sub(pattern, "NAME", self.text)
            tok_text = re.sub(r"<>.*?<\\\>", "OTHERS", tok_text)
            texts.append((tok_text, name))

        return texts

    def names(self):
        names = []
        pattern = r"<>(.*?)<\\>"
        for match in re.finditer(pattern, self.text):
            name = match.group(1)
            if name not in names:
                names.append(name)

        return names


def get_labeled(data_set):
    episodes = []
    lines = data_set.splitlines()

    for title, description, label in zip(lines[0::4], lines[1::4], lines[2::4]):
        if label == "T":
            ep = Episode(title, description, 0)
            episodes.append(ep)
        elif label == "G":
            ep = Episode(title, description, 1)
            episodes.append(ep)
        else:
            print("Wrong label!")
    n_total = len(lines)/4
    n_ep = len(episodes)
    if not n_total == n_ep:
        print("Only got " + str(n_ep/n_total*100) + "% of labeled data")

    return episodes


def get_unlabeled(data_set):
    episodes = []
    lines = data_set.splitlines()

    for title, description in zip(lines[0::3], lines[1::3]):
        ep = Episode(title, description)
        episodes.append(ep)

    return episodes

