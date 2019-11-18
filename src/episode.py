import spacy

import regex as re

nlp = spacy.load("en_core_web_sm")
# TODO: Obscure NAME on multple locations
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
        print(len(text.splitlines()))
        names = self.persons()

        for name in names:
            try:
                tok_text = re.sub(name, "NAME", text)
                print(name)
                print(len(tok_text.splitlines()))
            except:
                continue
            if tok_text != text:
                texts.append((tok_text, name))

        return texts

    def persons(self):
        """
        Returns a list of persons featured in the episode.
        """
        text = self.title + " \n " + self.description
        tokens = nlp(text)

        # Multiple names in a row is one name.
        names = []
        name = []
        for token in tokens:
            if token.ent_type_ == "PERSON" and token.text != "'s" and token.text != "'":
                token_text = re.sub("\n", "", token.text)
                name.append(token_text)
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

    return episodes


def get_unlabeled(data_set):
    episodes = []
    lines = data_set.splitlines()

    for title, description, label in zip(lines[0::3], lines[1::3]):
        ep = Episode(title, description)
        episodes.append(ep)

    return episodes

title = "Chapter 133: The Chase – First Day - Read by Kerry Shale - "
desc = "Introduced by Peter Donaldson, Recorded by Kate Bland - Cast Iron Studios, Edited and Mixed at dBs Music'I have written a blasphemous book', said Melville when his novel was first published in 1851, 'and I feel as spotless as the lamb'. Deeply subversive, in almost every way imaginable, Moby-Dick is a virtual, alternative bible - and as such, ripe for reinterpretation in this new world of new media. Out of Dominion was born its bastard child - or perhaps its immaculate conception - the Moby-Dick Big Read: an online version of Melville's magisterial tome: each of its 135 chapters read out aloud, by a mixture of the celebrated and the unknown, to be broadcast online, one new chapter each day, in a sequence of 135 downloads, publicly and freely accessible.Starting 16 September 2012!For more info please go to: www.mobydickbigread.com"

ep = Episode(title, desc)
ep.tokenize()
#print(ep.tokenize()[0])
#print(ep.tokenize()[1])
