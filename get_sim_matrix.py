import numpy as np
import spacy
from pathlib import Path

# load words
p = Path('words.txt')
with p.open('r') as f:
    words = [w.replace('\n', '') for w in f.readlines()]

# initialize matrix
num_words = len(words)
sim_matrix = np.zeros((num_words, num_words))

print(words)


# get glove model
nlp = spacy.load('en_core_web_lg')

# get glove vectors
tokens = nlp(' '.join(words))


for i, token1 in enumerate(tokens):
    for j, token2 in enumerate(tokens):
        sim = token1.similarity(token2)
        sim_matrix[i-1, j-1] = sim



print(sim_matrix)