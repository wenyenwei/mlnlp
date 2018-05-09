# Very basic article spinner for NLP class, which can be found at:
# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python

# Original Author: http://lazyprogrammer.me

# A very bad article spinner using trigrams.
from __future__ import print_function, division
from future.utils import iteritems


import nltk
import random
import numpy as np


# read dataset, kaggle twitter dataset https://www.kaggle.com/c/twitter-sentiment-analysis2/data
df = pd.read_csv('senti_test.csv')


# extract trigrams and insert into dictionary
trigrams = {}
for text in df['SentimentText']:
    s = text.lower()
    tokens = s.split()
    for i in range(len(tokens) - 2):
        k = (tokens[i], tokens[i+2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1])

# turn each array of middle-words into a probability vector
trigram_probabilities = {}
for k, words in iteritems(trigrams):
    # create a dictionary of word -> count
    if len(set(words)) > 1:
        # only do this when there are different possibilities for a middle word
        d = {}
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w, c in iteritems(d):
            d[w] = float(c) / n
        trigram_probabilities[k] = d


def random_sample(d):
    # choose a random sample from dictionary where values are the probabilities
    r = random.random()
    cumulative = 0
    for w, p in iteritems(d):
        cumulative += p
        if r < cumulative:
            return w


def test_spinner():
    text = random.choice(df['SentimentText'])
    s = text.lower()
    print("Original:", s)
    tokens = s.split()
    for i in range(len(tokens) - 2):
        if random.random() < 0.4: # 20% chance of replacement
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i+1] = w
    print("Spun:")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))


if __name__ == '__main__':
    test_spinner()