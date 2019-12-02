import regex as re
import random

class Episode:
    def __init__(self, title, description, guest=False):
        self.title = title
        self.description = description
        self.guest = guest
        self.text = title + '\n' + description
        #self.window = self.window(5)
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
            tok_text = re.sub(pattern, "FOCUSNAMEFOCUS", self.text)
            tok_text = re.sub(r"<>.*?<\\>", "NOTFOCUSNOT", tok_text)
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

    def window(self, size=5):
        text = self.title + " " + self.description
        words = text.split(" ")
        win = ["NIL" for i in range(size*2+1)]
        focus_word = re.findall(r"[^\r\n\t\f\v ]*FOCUSNAMEFOCUS[^\r\n\t\f\v ]*", text)
        try:
            focus_index = words.index(focus_word[0])
        except:
            self.skip = True
            print("Failed window")
            print(words)
            print(focus_word)
            input()
            return text

        word_index = focus_index - size
        for i, word in enumerate(win):
            if word_index < 0:
                word_index += 1
            elif word_index < len(words) - 1:
                win[i] = words[word_index]
                word_index +=1
        return " ".join(win)

        



def get_labeled(data_set, require_blance=False):
    episodes_T = []
    episodes_G = []
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
        else:
            print("Wrong label!")

    if require_blance:
        len_G = len(episodes_G)
        len_T = len(episodes_T)
        if len_G > len_T:
            episodes_G = episodes_G[:len_T]
        else:
            episodes_T = episodes_T[:len_G]

    episodes = []
    episodes.extend(episodes_G)
    episodes.extend(episodes_T)
    random.shuffle(episodes)

    return episodes


def get_unlabeled(data_set):
    episodes = []
    lines = data_set.splitlines()

    for title, description in zip(lines[0::3], lines[1::3]):
        ep = Episode(title, description)
        episodes.append(ep)

    return episodes

