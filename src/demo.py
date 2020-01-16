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
import numpy as np

"""
Used for demostrating the model by typing made up samples.
"""

model_file_name = sys.argv[1]

with open(model_file_name, 'rb') as f:
    model, tokenizer = pickle.load(f)

while True:
    win_size = 10
    text = input("Type an episode description: \n")
    tagged_text = tag_names(text) 
    ep = Episode("", tagged_text)
    for text, name in ep.tokenize():
        text = window(text, win_size)
        text = tokenizer.texts_to_sequences([text])
        text = pad_sequences(text, maxlen=2*win_size+1)
        y = model.predict(text)
        if y[0] > 0.5:
            print(name, "is a Guest", str(np.round(y[0][0]*100)), "%")
        else:
            print(name, "is a Topic", str(np.round((1-y[0][0])*100)), "%")
    print()
        

