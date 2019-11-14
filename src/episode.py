import spacy
import regex as re

nlp = spacy.load("en_core_web_sm")
#TODO: Obscure NAME on multple locations
class Episode():

    def __init__(self, title, description):
        self.title = title
        self.description = description

    def print(self):
        print(self.title)
        print(self.description)

    def window(self, win_size=3):
        padding = ['NIL' for n in range(win_size)]
        words = padding + self.description.split() + padding
        window = None
        for i, word in enumerate(words):
            if word == 'NAME':
                window = []
                window.extend(words[i-win_size:i])
                window.extend(words[i:i+win_size+1])
        return window

    def tokenize(self):
        """ 
        Returns a list of tokenized titles and descriptions, names names and pos,
        where the names are replaced with "NAME":
        """
        texts = []
        text = self.title + " " +  self.description
        names = self.persons()

        for name in names:
            try:
                tok_text = re.sub(name, 'NAME', text)
            except:
                continue
            if tok_text != text:
                texts.append((tok_text, name))

        return texts

    def persons(self):
        """
        Returns a list of persons featured in the episode.
        """
        text = (self.title + "\n" + self.description)
        tokens = nlp(text)
        
        # Multiple names in a row is one name.
        names = []
        name = []
        for token in tokens:
            if token.ent_type_ == 'PERSON' and token.text != "'s" and token.text != "'":
                name.append(token.text)
            elif len(name) > 1:
                complete_name = " ".join(name)
                if complete_name not in names:
                    names.append(complete_name)
                name = []
        complete_name = " ".join(name)
        if len(name) > 1 and complete_name not in names :
            names.append(" ".join(name))

        return names



        
        
        
