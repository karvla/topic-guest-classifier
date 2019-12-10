import pickle
import sys
import nltk
from nltk.tag.stanford import StanfordNERTagger
import sys
import regex as re
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tag_names import tag_names
from episode import Episode, window

"""
Used for testing a model in kersa.
"""

model_file_name = sys.argv[1]

with open(model_file_name, 'rb') as f:
    model, tokenizer = pickle.load(f)

while True:
    text = input("Type an episode description: \n")
    tagged_text = tag_names(text) 
    ep = Episode("", tagged_text)
    for text, name in ep.tokenize():
        text = window(text, 30)
        text = tokenizer.texts_to_sequences([text])
        text = pad_sequences(text, maxlen=61)
        y = model.predict(text)
        if y[0] > 0.5:
            print(name, "is a Guest", y[0])
        else:
            print(name, "is a Topic", y[0])
        

