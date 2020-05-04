"""main_3a.py

An instance of the Text class should be initialized with a file path (a file or
directory). The example here uses object as the super class, but you may use
nltk.text.Text as the super class.

An instance of the Vocabulary class should be initialized with an instance of
Text (not nltk.text.Text).

"""
import os
import math
import re
from itertools import count

import nltk
from nltk import collections
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist
from nltk.text import Text


class Text(nltk.text.Text):
    STOPLIST = set(nltk.corpus.stopwords.words())

    # Vocabulary with 234,377 English words from NLTK
    ENGLISH_VOCABULARY = set(w.lower() for w in nltk.corpus.words.words())

    # tokens within text
    tokens = nltk.word_tokenize(" ")

    # raw text
    raw_string = ""

    def __init__(self, path):
        if os.path.isfile(path):
            with open(path) as fh:
                self.raw_string = fh.read()

                self.tokens = nltk.word_tokenize(self.raw_string)

        elif os.path.isdir(path):
            # restrict to files with the mrg extension, avoiding hidden files like .DS_Store
            # that can cause trouble
            corpus = PlaintextCorpusReader(path, '.*.mrg')
            self.raw_string = corpus.raw()
            self.tokens = nltk.word_tokenize(self.raw_string)

    def find_sirs(self):
        sirs = []
        b = self.bigrams()
        for x, w in b:
            if x == "Sir":
                sirs.append(x + " " + w)
        return sirs

    def find_repeated_words(self):
        reapted_words = []
        brackets = re.findall(r"((\b\w\w\w+\b)\W)+(\1)", self.raw_string)
        for m in brackets:
            reapted_words.append((m[0] + m[1] + " " + m[2]).strip())
        return set(reapted_words)

    def find_brackets(self):
        brackets = re.findall(r"[^[]*\[([^]]*)\]", self.raw_string)
        parentheses = re.findall(r"\(([^\)]+)\)", self.raw_string)
        return parentheses + brackets

    def find_roles(self):
        m = re.findall("([A-Za-z0-9#].+(:))", self.raw_string)
        roles = []
        for x in m:
            if "SCENE" not in x[0]:
                roles.append(x[0].replace(":", ""))
        return set(roles)

    def readability(self, method):
        pass

    def type_count(text):
        """Returns the type count, with minimal normalization by lower casing."""
        # an alternative would be to use the method nltk.text.Text.vocab()
        return len(set([w.lower() for w in text.tokens]))

    def token_count(text):
        """Just return all tokens."""
        return len(text.tokens)

    def sentence_count(text):
        """Return number of sentences, using the simplistic measure of counting period,
        exclamation marks and question marks."""
        return len([t for t in text if t in ('.', '!', '?')])

    def is_content_word(self, word):
        """A content word is not on the stoplist and its first character is a letter."""
        return word.lower() not in self.STOPLIST and word[0].isalpha()

    def most_frequent_content_words(text):
        """Return a list with the 25 most frequent content words and their
        frequencies. The list has (word, frequency) pairs and is ordered on the
        frequency."""
        dist = nltk.FreqDist([w for w in text if text.is_content_word(w)])
        return dist.most_common(n=25)

    def most_frequent_bigrams(self):
        """Return a list with the 25 most frequent bigrams that only contain
        content words. The list returned should have pairs where the first
        element in the pair is the bigram and the second the frequency, as in
        ((word1, word2), frequency), these should be ordered on frequency."""
        filtered_bigrams = [b for b in list(nltk.bigrams(self.tokens))
                            if self.is_content_word(b[0]) and self.is_content_word(b[1])]
        dist = nltk.FreqDist([b for b in filtered_bigrams])
        return dist.most_common(n=25)

    def bigrams(self):

        filtered_bigrams = [b for b in list(nltk.bigrams(self.tokens))
                            if self.is_content_word(b[0]) and self.is_content_word(b[1])]
        dist = nltk.FreqDist([b for b in filtered_bigrams])
        return dist


class Vocabulary(Text):
    """Class to store all information on a vocabulary, where a vocabulary is created
    from a text. The vocabulary includes the text, a frequency distribution over
    that text, the vocabulary items themselves (as a set) and the sizes of the
    vocabulary and the text. We do not store POS and gloss, for those we rely on
    WordNet. The vocabulary is contrained to those words that occur in a
    standard word list. Vocabulary items are not normalized, except for being in
    lower case."""

    def __init__(self, text):

        self.text = text.tokens
        # keeping the unfiltered list around for statistics
        self.all_items = set([w.lower() for w in text])
        self.items = self.all_items.intersection(text.ENGLISH_VOCABULARY)
        # restricting the frequency dictionary to vocabulary items
        self.fdist = nltk.FreqDist(t.lower() for t in text if t.lower() in self.items)
        self.text_size = len(self.text)
        self.vocab_size = len(self.items)

    def __str__(self):
        return "<Vocabulary size=%d text_size=%d>" % (self.vocab_size, self.text_size)

    def __len__(self):
        return self.vocab_size

    def frequency(self, word):
        return self.fdist[word]

    def pos(self, word):
        # do not volunteer the pos for words not in the vocabulary
        if word not in self.items:
            return None
        synsets = wn.synsets(word)
        return synsets[0].pos() if synsets else 'n'

    def gloss(self, word):
        # do not volunteer the gloss (definition) for words not in the vocabulary
        if word not in self.items:
            return None
        synsets = wn.synsets(word)
        # make a difference between None for words not in vocabulary and words
        # in the vocabulary that do not have a gloss in WordNet
        return synsets[0].definition() if synsets else 'NO DEFINITION'

    def kwic(self, word):
        nltk_text = nltk.Text(self.text)
        return nltk_text.concordance(word)
