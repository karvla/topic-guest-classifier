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
from episode import Episode

"""
Used for testing a model in kersa.
"""

model_file_name = sys.argv[1]

with open(model_file_name, 'rb') as f:
    model, tokenizer = pickle.load(f)

while True:
    print()
    text = input("Type an episode description: \n")
    tagged_text = tag_names(text) 
    ep = Episode("", tagged_text)
    for text, name in ep.tokenize():
        text = tokenizer.texts_to_sequences([text])
        text = pad_sequences(text, maxlen=60)
        y = model.predict(text)
        if y > 0.5:
            print(name, "is a guest", y[0][0])
        else:
            print(name, "is a topic", y[0][0])
        

