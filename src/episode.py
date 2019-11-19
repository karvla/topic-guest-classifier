import nltk
import spacy
from nltk.tag.stanford import StanfordNERTagger

st = StanfordNERTagger('/home/karvla/projects/topic-guest-classifier/english.all.3class.distsim.crf.ser.gz', '/home/karvla/projects/topic-guest-classifier/stanford-ner.jar')

import regex as re

nlp = spacy.load("en_core_web_sm")
class Episode:
    def __init__(self, title, description, guest=False):
        self.title = title
        self.description = description
        self.guest = guest

    def _text(self):
        return self.title + " " + self.description

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
        text = self.title + " \n " + self.description
        names = self.persons()

        for name in names:
            try:
                tok_text = re.sub(name, "NAME", text)
            except:
                continue
            if tok_text != text:
                texts.append((tok_text, name))

        return texts

    def persons(self):
        """
        Returns a list of persons featured in the episode.
        """
        text = nltk.tokenize.word_tokenize(self.title + " \n " + self.description)
        # Multiple names in a row is one name.
        names = []
        name = []
        
        for word, tag in st.tag(text):
            if tag == "PERSON" and word != "'s" and word != "'":
                word = re.sub("\n", "", word)
                name.append(word)
            elif len(name) > 1:
                complete_name = " ".join(name)
                if complete_name not in names:
                    names.append(complete_name)
                name = []
        complete_name = " ".join(name)
        if len(name) > 1 and complete_name not in names:
            names.append(" ".join(name))

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

    for title, description, label in zip(lines[0::3], lines[1::3]):
        ep = Episode(title, description)
        episodes.append(ep)

    return episodes
