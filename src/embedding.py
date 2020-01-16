import sys
import pickle
import numpy as np

"""
Creates an embeddings index and saves it.
"""
glove_file_path = sys.argv[1]

embeddings_index = {}
with open(glove_file_path, "r") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

with open("./data/embeddings_index.pickle", 'wb') as f:
    pickle.dump(embeddings_index, f)
