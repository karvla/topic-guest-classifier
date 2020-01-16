import regex as re
import random


class Episode:
    def __init__(self, title, description, guest=None):
        self.title = title
        self.description = description
        #self.text = title + "\n" + description
        self.guest = guest
        self.skip = False

    def tokenize(self):
        """ 
        Returns a list of tokenized titles and descriptions, names names and pos,
        where the names are replaced with "NAME":
        """
        texts = []
        names = self.names()

        for name in names:
            pattern = r"<>" + name + r"<\\>"
            tok_text = re.sub(pattern, "FOCUSNAMEFOCUS", self.description)
            tok_text = re.sub(r"<>.*?<\\>", "NOTFOCUSNOT", tok_text)
            texts.append((tok_text, name))

        return texts

    def focus_sentence(self):
        pattern = r"[^.!\?]* FOCUSNAMEFOCUS[^.]*\."
        result = re.findall(pattern, self.description, re.IGNORECASE | re.MULTILINE)
        if result:
            return result[0]
        else:
            return window(self.description, 30)

    def names(self):
        names = []
        pattern = r"<>(.*?)<\\>"
        for match in re.finditer(pattern, self.description):
            name = match.group(1)
            if name not in names:
                names.append(name)

        return names


def window(text, size=30):
    words = text.split(" ")
    win = ["NIL" for i in range(size * 2 + 1)]
    focus_index = size
    for n, word in enumerate(words):
        focus_word = re.findall(r"FOCUSNAMEFOCUS", word)
        if focus_word:
            focus_index = n
            break

    
    word_index = focus_index - size

    for i, word in enumerate(win):
        if word_index < 0:
            word_index += 1
        elif word_index < len(words) - 1:
            win[i] = words[word_index]
            word_index += 1
    return " ".join(win)


def get_labeled(data_set, require_blance=False):
    episodes_T = []
    episodes_G = []
    episodes_unlabeled = []
    lines = data_set.splitlines()

    for title, description, label in zip(lines[0::4], lines[1::4], lines[2::4]):
        if label == "T":
            ep = Episode(title, description, 0)
            if ep.skip:
                continue
            episodes_T.append(ep)

        elif label == "G":
            ep = Episode(title, description, 1)
            if ep.skip:
                continue
            episodes_G.append(ep)

        elif label == "_":
            ep = Episode(title, description)
            episodes_unlabeled.append(ep)

        else:
            print("Wrong label!")


    if require_blance:
        len_G = len(episodes_G)
        len_T = len(episodes_T)
        if len_G > len_T:
            episodes_G = episodes_G[:len_T]
        else:
            episodes_T = episodes_T[:len_G]

    labeled_episodes = episodes_G + episodes_T
    random.shuffle(labeled_episodes)

    return labeled_episodes, episodes_unlabeled


def get_unlabeled(data_set, size=3):
    episodes = []
    lines = data_set.splitlines()

    for title, description in zip(lines[0::size], lines[1::size]):
        ep = Episode(title, description)
        episodes.append(ep)

    return episodes
