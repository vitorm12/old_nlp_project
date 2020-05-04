"""postag.py

Custom POS tagger

"""

import nltk
from nltk.corpus import brown
from pickle import dump
from pickle import load


class POSTagger(object):

    def __init__(self):
        brown_sentences = brown.tagged_sents(categories='news')
        size = int(len(brown_sentences) * 0.9)
        self.train_sentences = brown_sentences[:size]

    def open_test_model(self,category):
        open_model = open('t.pkl', 'rb')
        tags = load(open_model)
        open_model.close()
        return tags.evaluate(brown.tagged_sents(categories=category))

    def create_trainer(self):
        t0 = nltk.DefaultTagger('NN')
        t1 = nltk.UnigramTagger(self.train_sentences, backoff=t0)  #
        t2 = nltk.BigramTagger(self.train_sentences, backoff=t1)
        t3 = nltk.TrigramTagger(self.train_sentences, backoff=t2)
        output = open('t.pkl', 'wb')
        dump(t3, output, -1)
        output.close()

    def begin_model_execute(self, sentence):
        input_model = open('t.pkl', 'rb')
        tagger = load(input_model)
        input_model.close()
        tokens = sentence.split()
        return tagger.tag(tokens)


